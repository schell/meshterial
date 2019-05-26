use vulkano::VulkanObject;
use vulkano::image::swapchain::SwapchainImage;
use vulkano::buffer::cpu_pool::CpuBufferPool;
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::device::{Device, Queue};
use vulkano::instance::{Instance, RawInstanceExtensions, PhysicalDevice};
use vulkano::image::attachment::AttachmentImage;
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract};
use vulkano::pipeline::viewport::Viewport;
use vulkano::image::ImmutableImage;
use image::GenericImageView;
use vulkano::swapchain;
use vulkano::swapchain::{
  PresentMode,
  Surface,
  SurfaceTransform,
  Swapchain,
  AcquireError,
  SwapchainAcquireFuture,
  SwapchainCreationError,
};
use vulkano::image::Dimensions;
use vulkano::format::Format;
use vulkano::sync::now;
use vulkano::sync::GpuFuture;

use sdl2::Sdl;
use sdl2::video::{WindowContext, Window};

use nalgebra::{Matrix4, Vector2};
use nalgebra_glm as glm;

use std::ffi::CString;
use std::sync::Arc;
use std::rc::Rc;
use std::mem;
use std::collections::HashMap;

mod sendable;
use self::sendable::Sendable;


/// The VkRenderer takes care of making the sdl2 context, choosing the vulkan
/// instance, device, queue, etc. Basically everything except the pipeline.
/// Pipelines are created separately.
pub struct VkRenderer {
  pub ctx: Sdl,
  pub window: Window,
  pub instance: Arc<Instance>,
  pub device: Arc<Device>,
  pub queue: Arc<Queue>,
  pub dimensions: [u32; 2],
  pub surface: Arc<Surface<Sendable<Rc<WindowContext>>>>,
  pub swapchain: Arc<Swapchain<Sendable<Rc<WindowContext>>>>,
  pub images: Vec<Arc<SwapchainImage<Sendable<Rc<WindowContext>>>>>,
  pub framebuffers: Option<Vec<Arc<FramebufferAbstract + Send + Sync>>>,

  pub render_pass: Arc<RenderPassAbstract + Send + Sync>,

  pub recreate_swapchain: bool,
  pub previous_frame_end: Option<Box<GpuFuture>>,
  pub image_num: Option<usize>,
  pub acquire_future: Option<SwapchainAcquireFuture<Sendable<Rc<WindowContext>>>>,
  pub dynamic_state: DynamicState,

  pub command_buffer_builder: Option<AutoCommandBufferBuilder>,

  pub proj_buffer_pool: CpuBufferPool<Matrix4<f32>>,

  /// A store of loaded textures.
  texture_store: HashMap<String, (Arc<ImmutableImage<Format>>, Vector2<u32>)>,
}


impl VkRenderer {
  /// Create and return a Vulkan instance.
  pub fn instance(window: &Window) -> Arc<Instance> {
    // When we create an instance, we have to pass a list of extensions that we want to enable.
    //
    // All the window-drawing functionalities are part of non-core extensions that we need
    // to enable manually. To do so, we ask the `vulkano_win` crate for the list of extensions
    // required to draw to a window.
    let instance_extensions = window.vulkan_instance_extensions().unwrap();
    let raw_instance_extensions = RawInstanceExtensions::new(instance_extensions.iter().map(
      |&v| CString::new(v).unwrap()
    ));

    // Now creating the instance.
    Instance::new(None, raw_instance_extensions, None).expect("failed to create Vulkan instance")
  }


  /// Create and return a new vulkan based renderer.
  pub fn new() -> VkRenderer {
    let ctx = sdl2::init().unwrap();
    let window = ctx
      .video().unwrap()
      .window("Window", 800, 600)
      .resizable()
      .vulkan()
      .build().unwrap();

    // The first step of any vulkan program is to create an instance.
    let instance = VkRenderer::instance(&window);

    // We then choose which physical device to use.
    //
    // In a real application, there are three things to take into consideration:
    //
    // - Some devices may not support some of the optional features that may be required by your
    //   application. You should filter out the devices that don't support your app.
    //
    // - Not all devices can draw to a certain surface. Once you create your window, you have to
    //   choose a device that is capable of drawing to it.
    //
    // - You probably want to leave the choice between the remaining devices to the user.
    //
    // For the sake of the example we are just going to use the first device, which should work
    // most of the time.
    let physical_instance = instance.clone();
    let physical =
      PhysicalDevice::enumerate(&physical_instance)
        .next()
        .expect("no physical device available");
    // Some little debug infos.
    println!("Using device: {} (type: {:?})", physical.name(), physical.ty());

    // The objective of this example is to draw a triangle on a window. To do so, we first need to
    // create the window.
    //
    // TODO: This text is wrong.
    // This is done by creating a `Window` from the `sdl2` crate, then calling the
    // `build_vk_surface` method provided by the `VkSurfaceBuild` trait from `vulkano_win`. If you
    // ever get an error about `build_vk_surface` being undefined in one of your projects, this
    // probably means that you forgot to import this trait.
    //
    // This returns a `vulkano::swapchain::Surface` object that contains both a cross-platform winit
    // window and a cross-platform Vulkan surface that represents the surface of the window.
    let surface_handle = window.vulkan_create_surface(instance.internal_object()).unwrap();
    let window_context = Sendable::new(window.context());
    let surface = Arc::new(unsafe { Surface::from_raw_surface(instance.clone(), surface_handle, window_context) });

    // The next step is to choose which GPU queue will execute our draw commands.
    //
    // Devices can provide multiple queues to run commands in parallel (for example a draw queue
    // and a compute queue), similar to CPU threads. This is something you have to have to manage
    // manually in Vulkan.
    //
    // In a real-life application, we would probably use at least a graphics queue and a transfers
    // queue to handle data transfers in parallel. In this example we only use one queue.
    //
    // We have to choose which queues to use early on, because we will need this info very soon.
    let queue = physical.queue_families().find(|&q| {
      // We take the first queue that supports drawing to our window.
      q.supports_graphics() && surface.is_supported(q).unwrap_or(false)
    }).expect("couldn't find a graphical queue family");

    // Now initializing the device. This is probably the most important object of Vulkan.
    //
    // We have to pass five parameters when creating a device:
    //
    // - Which physical device to connect to.
    //
    // - A list of optional features and extensions that our program needs to work correctly.
    //   Some parts of the Vulkan specs are optional and must be enabled manually at device
    //   creation. In this example the only thing we are going to need is the `khr_swapchain`
    //   extension that allows us to draw to a window.
    //
    // - A list of layers to enable. This is very niche, and you will usually pass `None`.
    //
    // - The list of queues that we are going to use. The exact parameter is an iterator whose
    //   items are `(Queue, f32)` where the floating-point represents the priority of the queue
    //   between 0.0 and 1.0. The priority of the queue is a hint to the implementation about how
    //   much it should prioritize queues between one another.
    //
    // The list of created queues is returned by the function alongside with the device.
    let (device, mut queues) = {
      let device_ext = vulkano::device::DeviceExtensions {
        khr_swapchain: true,
        .. vulkano::device::DeviceExtensions::none()
      };

      Device::new(physical, physical.supported_features(), &device_ext,
                  [(queue, 0.5)].iter().cloned()).expect("failed to create device")
    };

    // Since we can request multiple queues, the `queues` variable is in fact an iterator. In this
    // example we use only one queue, so we just retreive the first and only element of the
    // iterator and throw it away.
    let queue = queues.next().unwrap();

    // Querying the capabilities of the surface. When we create the swapchain we can only
    // pass values that are allowed by the capabilities.
    let caps = surface.capabilities(physical)
      .expect("failed to get surface capabilities");

    // The dimensions of the surface.
    // This variable needs to be mutable since the viewport can change size.
    let dimensions = caps.current_extent.unwrap_or([1024, 768]);

    // We choose the dimensions of the swapchain to match the current extent of the surface.
    // If `caps.current_extent` is `None`, this means that the window size will be determined
    // by the dimensions of the swapchain, in which case we just use the width and height defined above.

    // The alpha mode indicates how the alpha value of the final image will behave. For example
    // you can choose whether the window will be opaque or transparent.
    let alpha = caps.supported_composite_alpha.iter().next().unwrap();

    // Choosing the internal format that the images will have.
    let format = caps.supported_formats[0].0;
    println!("Choosing format {:?} from {:?}", format, caps.supported_formats);

    // Please take a look at the docs for the meaning of the parameters we didn't mention.
    let (swapchain, images) =
      Swapchain::new(
        device.clone(),
        surface.clone(),
        caps.min_image_count,
        format,

        dimensions,
        1,
        caps.supported_usage_flags,
        &queue,

        SurfaceTransform::Identity,
        alpha,

        // See https://github.com/vulkano-rs/vulkano/issues/252
        PresentMode::Immediate,
        true,

        None
      ).expect("failed to create swapchain");

    // Before we can draw on the surface, we have to create what is called a swapchain. Creating
    // a swapchain allocates the color buffers that will contain the image that will ultimately
    // be visible on the screen. These images are returned alongside with the swapchain.

    // The render pass we created above only describes the layout of our framebuffers. Before we
    // can draw we also need to create the actual framebuffers.
    //
    // Since we need to draw to multiple images, we are going to create a different framebuffer for
    // each image.
    let framebuffers = None;

    // Here we pull in our shaders from the shader module.

    // At this point, OpenGL initialization would be finished. However in Vulkan it is not. OpenGL
    // implicitly does a lot of computation whenever you draw. In Vulkan, you have to do all this
    // manually.

    // The next step is to create a *render pass*, which is an object that describes where the
    // output of the graphics pipeline will go. It describes the layout of the images
    // where the colors, depth and/or stencil information will be written.
    let render_pass = Arc::new(
      single_pass_renderpass!(
        device.clone(),
        attachments: {
          // `color` is a custom name we give to the first and only attachment.
          color: {
            // `load: Clear` means that we ask the GPU to clear the content of this
            // attachment at the start of the drawing.
            load: Clear,
            // `store: Store` means that we ask the GPU to store the output of the draw
            // in the actual image. We could also ask it to discard the result.
            store: Store,
            // `format: <ty>` indicates the type of the format of the image. This has to
            // be one of the types of the `vulkano::format` module (or alternatively one
            // of your structs that implements the `FormatDesc` trait). Here we use the
            // generic `vulkano::format::Format` enum because we don't know the format in
            // advance.
            format: swapchain.format(),
            // TODO:
            samples: 1,
          },
          depth: {
            load: Clear,
            store: DontCare,
            format: Format::D16Unorm,
            samples: 1,
          }
        },
        pass: {
          // We use the attachment named `color` as the one and only color attachment.
          color: [color],
          depth_stencil: {depth}
        }
      ).unwrap()
    );

        // In some situations, the swapchain will become invalid by itself. This includes for example
    // when the window is resized (as the images of the swapchain will no longer match the
    // window's) or, on Android, when the application went to the background and goes back to the
    // foreground.
    //
    // In this situation, acquiring a swapchain image or presenting it will return an error.
    // Rendering to an image of that swapchain will not produce any error, but may or may not work.
    // To continue rendering, we need to recreate the swapchain by creating a new swapchain.
    // Here, we remember that we need to do this for the next loop iteration.
    let recreate_swapchain = false;

    let dynamic_state = DynamicState {
      line_width: None,
      viewports: Some(vec![Viewport {
        origin: [0.0, 0.0],
        dimensions: [dimensions[0] as f32, dimensions[1] as f32],
        depth_range: 0.0 .. 1.0,
      }]),
      scissors: None,
    };


    let proj_buffer_pool = CpuBufferPool::upload(device.clone());

    VkRenderer {
      ctx,
      window,
      instance,
      device,
      queue,
      dimensions,
      surface,
      swapchain,
      images,
      framebuffers,
      render_pass,
      recreate_swapchain,
      dynamic_state,

      previous_frame_end: None,
      image_num: None,
      acquire_future: None,
      command_buffer_builder: None,

      proj_buffer_pool,

      texture_store: HashMap::new()
    }
  }


  pub fn recreate_swapchain(&mut self) -> Result<(), SwapchainCreationError> {
    let physical =
      PhysicalDevice::enumerate(&self.instance)
      .next()
      .expect("no device available");
    // Get the new dimensions for the viewport/framebuffers.
    self.dimensions = self.surface.capabilities(physical)
      .expect("failed to get surface capabilities")
      .current_extent.unwrap();

    // Update the dynamic_state with the new dimensions.
    self.dynamic_state.viewports = Some(vec![Viewport {
      origin: [0.0, 0.0],
      dimensions: [
        self.dimensions[0] as f32,
        self.dimensions[1] as f32
      ],
      depth_range: 0.0 .. 1.0,
    }]);

    match self.swapchain.recreate_with_dimension(self.dimensions) {
      Ok((new_swapchain, new_images)) => {
        mem::replace(&mut self.swapchain, new_swapchain);
        mem::replace(&mut self.images, new_images);
        self.framebuffers = None;
        Ok(())
      }
      Err(e) => Err(e)
    }
  }


  pub fn with_command_builder<T> (&mut self, add_cmds:T)
    where T: FnOnce(AutoCommandBufferBuilder) -> AutoCommandBufferBuilder
  {
    self.command_buffer_builder = Some(add_cmds(
      if let Some(builder) = self.command_buffer_builder.take() {
        builder
      } else {
        AutoCommandBufferBuilder::primary_one_time_submit(self.device.clone(), self.queue.family())
          .expect("Could not create a new command buffer builder.")
      }
    ))
  }


  /// End the last frame of rendering and begin the next.
  /// Sets up the command builder.
  /// If Some(true) is returned it means that the framebuffers were resized.
  pub fn start_next_frame(&mut self) -> Option<bool> {
    let mut resized = false;
    // It is important to call this function from time to time, otherwise resources will keep
    // accumulating and you will eventually reach an out of memory error.
    // Calling this function polls various fences in order to determine what the GPU has
    // already processed, and frees the resources that are no longer needed.
    match self.previous_frame_end.as_mut() {
      Some(p) => {p.cleanup_finished();}
      None => {}
    }

    // Destroying the `GpuFuture` blocks until the GPU is finished executing it. In order to avoid
    // that, we store the submission of the previous frame here.
    if self.previous_frame_end.is_none() {
      self.previous_frame_end = Some(Box::new(now(self.device.clone())) as Box<GpuFuture>);
    }

    // If the swapchain needs to be recreated, recreate it
    if self.recreate_swapchain {
      match self.recreate_swapchain() {
        Ok(()) => {}
        // This error tends to happen when the user is manually resizing the window.
        // Simply restarting the loop is the easiest way to fix this issue.
        Err(SwapchainCreationError::UnsupportedDimensions) => {
          return None;
        }
        Err(err) => panic!("{:?}", err)
      }
      self.recreate_swapchain = false;
    }

    // Because framebuffers contains an Arc on the old swapchain, we need to
    // recreate framebuffers as well.
    if self.framebuffers.is_none() {
      let depth_buffer = AttachmentImage::transient(
        self.device.clone(),
        self.dimensions,
        Format::D16Unorm
      ).unwrap();
      // Make new framebuffers
      let new_framebuffers:Option<Vec<Arc<_>>> = Some(self.images.iter().map(|image| {
        let afb = Arc::new(
          Framebuffer::start(self.render_pass.clone())
            .add(image.clone()).expect("Could not add image to framebuffer.")
            .add(depth_buffer.clone()).expect("Could not add depth buffer to framebuffer.")
            .build().expect("Could not build new framebuffer.")
        );
        afb as Arc<FramebufferAbstract + Send + Sync>
      }).collect::<Vec<_>>());
      mem::replace(&mut self.framebuffers, new_framebuffers);

      resized = true;
    }

    // Before we can draw on the output, we have to *acquire* an image from the swapchain. If
    // no image is available (which happens if you submit draw commands too quickly), then the
    // function will block.
    // This operation returns the index of the image that we are allowed to draw upon.
    //
    // This function can block if no image is available. The parameter is an optional timeout
    // after which the function call will return an error.
    let swap_next = swapchain::acquire_next_image(self.swapchain.clone(), None);
    let (image_num, acquire_future) = match swap_next {
      Ok(r) => r,
      Err(AcquireError::OutOfDate) => {
        self.recreate_swapchain = true;
        return None;
      },
      Err(err) => panic!("{:?}", err)
    };

    self.image_num = Some(image_num);
    self.acquire_future = Some(acquire_future);

    self.with_command_builder(|cmds| cmds);

    Some(resized)
  }


  /// Starts a rendering. If None is returned, skip rendering this frame and pick
  /// it back up later. If Some(true) is returned it means that the framebuffers were resized.
  pub fn begin_rendering(&mut self) {
    // In order to draw, we have to build a *command buffer*. The command buffer object holds
    // the list of commands that are going to be executed.
    //
    // Building a command buffer is an expensive operation (usually a few hundred
    // microseconds), but it is known to be a hot path in the driver and is expected to be
    // optimized.
    //
    // Note that we have to pass a queue family when we create the command buffer. The command
    // buffer will only be executable on that given queue family.
    let image_num = self
      .image_num
      .expect("Could not get image_num - maybe 'start_next_frame' has not been called.");
    let fbs = self
      .framebuffers
      .as_ref()
      .expect("Could not get framebuffers as a ref.")[image_num]
      .clone();
    self.with_command_builder(|cmds| {
      cmds
      // Before we can draw, we have to *enter a render pass*. There are two methods to do
      // this: `draw_inline` and `draw_secondary`. The latter is a bit more advanced and is
      // not covered here.
      //
      // The third parameter builds the list of values to clear the attachments with. The API
      // is similar to the list of attachments when building the framebuffers, except that
      // only the attachments that use `load: Clear` appear in the list.
        .begin_render_pass(
          fbs,
          false,
          vec![
            [0.0, 0.0, 1.0, 1.0].into(),
            1f32.into()
          ]
        ).expect("Could not begin_render_pass.")
    });
  }


  /// Commit a buffer builder to be rendered.
  pub fn commit_rendering(&mut self) {
    // We leave the render pass by calling `end_render_pass`. Note that if we had multiple
    // subpasses we could have called `next_inline` (or `next_secondary`) to jump to the
    // next subpass.
    let command_buffer = self
      .command_buffer_builder
      .take().expect("Could not take command_buffer - maybe 'begin_rendering' was not called.")
      .end_render_pass().expect("Could not end_render_pass.")
      .build().expect("Could not build the command_buffer.");

    let future = self
      .previous_frame_end
      .take().expect("Could not take previous_frame_end.")
      .join(
        self
          .acquire_future
          .take()
          .expect("Could not get acquired_future - maybe 'begin_rendering' was not called before 'commit_rendering'.")
      )
      .then_execute(self.queue.clone(), command_buffer)
      .expect("Could not expecute the command_buffer.")
      // The color output is now expected to contain our geometry. But in order to show it on
      // the screen, we have to *present* the image by calling `present`.
      //
      // This function does not actually present the image immediately. Instead it submits a
      // present command at the end of the queue. This means that it will only be presented once
      // the GPU has finished executing the command buffer that draws the triangle.
      .then_swapchain_present(
        self.queue.clone(),
        self.swapchain.clone(),
        self
          .image_num
          .take()
          .expect("Could not get image_num - maybe 'begin_rendering' was not called before 'commit_rendering'.")
      )
      .then_signal_fence_and_flush();

    match future {
      Ok(future) => {
        self.previous_frame_end = Some(Box::new(future) as Box<_>);
      }
      Err(vulkano::sync::FlushError::OutOfDate) => {
        self.recreate_swapchain = true;
        self.previous_frame_end = Some(Box::new(vulkano::sync::now(self.device.clone())) as Box<_>);
      }
      Err(e) => {
        println!("{:?}", e);
        self.previous_frame_end = Some(Box::new(vulkano::sync::now(self.device.clone())) as Box<_>);
      }
    }

    // Note that in more complex programs it is likely that one of `acquire_next_image`,
    // `command_buffer::submit`, or `present` will block for some time. This happens when the
    // GPU's queue is full and the driver has to wait until the GPU finished some work.
    //
    // Unfortunately the Vulkan API doesn't provide any way to not wait or to detect when a
    // wait would happen. Blocking may be the desired behavior, but if you don't want to
    // block you should spawn a separate thread dedicated to submissions.
  }


  /// Returns a loaded image and its size.
  pub fn load_texture (
    &mut self,
    path: &String,
  ) -> (Arc<ImmutableImage<Format>>, Vector2<u32>) {
    if self.texture_store.contains_key(path) {
      let stuff = self.texture_store
        .get(path)
        .expect("This should never happen");
      (stuff.0.clone(), stuff.1)
    } else {
      let dyn_img = image::open(path)
        .expect(&format!("Could not open image '{}'", path));
      let width = dyn_img.width();
      let height = dyn_img.height();
      println!("Loaded image '{}' is color type {:?}", path, dyn_img.color());
      let data = dyn_img
        .to_bgra()
        .into_raw()
        .clone();

      let (texture, tex_future) = ImmutableImage::from_iter(
        data.iter().cloned(),
        Dimensions::Dim2d {
          width,
          height
        },
        Format::B8G8R8A8Unorm,
        self.queue.clone()
      ).expect("Could not create an immutable image.");

      let prev_future = self.previous_frame_end.take();
      if let Some(future) = prev_future {
        self.previous_frame_end = Some(Box::new(future.join(tex_future)));
      } else {
        self.previous_frame_end = Some(Box::new(tex_future));
      }

      let dims = glm::vec2(width, height);
      self.texture_store.insert(path.clone(), (texture.clone(), dims));
      (texture, dims)
    }
  }}
