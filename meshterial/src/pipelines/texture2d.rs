use vulkano::buffer::BufferUsage;
use vulkano::buffer::device_local::DeviceLocalBuffer;
use vulkano::descriptor::descriptor_set::{
  PersistentDescriptorSet,
  FixedSizeDescriptorSetsPool,
  DescriptorSet
};
use vulkano::device::Device;
use vulkano::instance::PhysicalDevice;
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::framebuffer::{RenderPassAbstract, Subpass};
use vulkano::image::immutable::ImmutableImage;
use vulkano::sampler::{Sampler, Filter, MipmapMode, SamplerAddressMode};
use vulkano::format::Format;

use nalgebra::*;

use std::sync::Arc;


mod vs {
  vulkano_shaders::shader!{
    ty: "vertex",
    path: "src/shaders/texture2d/vert.glsl"
  }
}

mod fs {
  vulkano_shaders::shader!{
    ty: "fragment",
    path: "src/shaders/texture2d/frag.glsl"
  }
}


#[derive(Debug, Clone)]
pub struct VertexUV {
  pub position: [f32; 2],
  pub uv: [f32; 2],
}
impl_vertex!(VertexUV, position, uv);


/// A graphics pipeline capable of rendering 2d textured geometry.
pub struct Texture2DPipeline {
  pub pipeline: Arc<GraphicsPipelineAbstract + Send + Sync>,
  pub proj_uniform_device_buffer:Arc<DeviceLocalBuffer<Matrix4<f32>>>,
  pub proj_desc_set: Arc<DescriptorSet + Send + Sync>,
  pub image_sampler_desc_pool: FixedSizeDescriptorSetsPool<Arc<GraphicsPipelineAbstract + Send + Sync>>,
  pub may_tex_desc_set: Option<Arc<DescriptorSet + Send + Sync>>,
}


impl Texture2DPipeline {
  pub fn new(
    render_pass: Arc<RenderPassAbstract + Send + Sync>,
    device: Arc<Device>,
    physical: PhysicalDevice
  ) -> Texture2DPipeline {
    let vs = vs::Shader::load(device.clone()).expect("failed to create shader module");
    let fs = fs::Shader::load(device.clone()).expect("failed to create shader module");
    let pipeline = Arc::new(
      GraphicsPipeline::start()
      // We need to indicate the layout of the vertices.
      // The type `SingleBufferDefinition` actually contains a template parameter corresponding
      // to the type of each vertex. But in this code it is automatically inferred.
        .vertex_input_single_buffer::<VertexUV>()
      // A Vulkan shader can in theory contain multiple entry points, so we have to specify
      // which one. The `main` word of `main_entry_point` actually corresponds to the name of
      // the entry point.
        .vertex_shader(vs.main_entry_point(), ())
      // The content of the vertex buffer describes a list of triangles.
        .triangle_list()
      // Use a resizable viewport set to draw over the entire window
        .viewports_dynamic_scissors_irrelevant(1)
      // See `vertex_shader`.
        .fragment_shader(fs.main_entry_point(), ())
        .depth_write(false)
      // Does it blend?
        .blend_alpha_blending()
      // We have to indicate which subpass of which render pass this pipeline is going to be used
      // in. The pipeline will only be usable from this particular subpass.
        .render_pass(Subpass::from(render_pass, 0).unwrap())
      // Now that our builder is filled, we call `build()` to obtain an actual pipeline.
        .build(device.clone())
        .unwrap()
    );

    // Uniform stuff
    let proj_uniform_device_buffer:Arc<DeviceLocalBuffer<_>> = DeviceLocalBuffer::new(
      device.clone(),
      BufferUsage::uniform_buffer_transfer_destination(),
      physical.queue_families()
    ).expect("Could not create uniform_device_buffer.");

    let proj_desc_set = Arc::new(
      PersistentDescriptorSet::start(pipeline.clone(), 0)
        .add_buffer(proj_uniform_device_buffer.clone()).expect("Could not add uniform_device_buffer.")
        .build()
        .expect("Could not build uniform_desc_set.")
    );

    let image_sampler_desc_pool =
      FixedSizeDescriptorSetsPool::new(pipeline.clone() as Arc<GraphicsPipelineAbstract + Send + Sync>, 1);

    Texture2DPipeline {
      pipeline,
      proj_uniform_device_buffer,
      proj_desc_set,
      image_sampler_desc_pool,
      may_tex_desc_set: None
    }
  }


  pub fn desc_set_for_texture (
    &mut self,
    texture: Arc<ImmutableImage<Format>>,
    device: Arc<Device>,
  ) -> Arc<DescriptorSet + Send + Sync> {
    let sampler = Sampler::new(
      device,
      Filter::Linear, Filter::Linear,
      MipmapMode::Nearest,
      SamplerAddressMode::Repeat, SamplerAddressMode::Repeat, SamplerAddressMode::Repeat,
      0.0,
      1.0,
      0.0, 0.0
    ).unwrap();

    Arc::new(
      self
        .image_sampler_desc_pool
        .next()
        .add_sampled_image(texture.clone(), sampler.clone()).expect("Could not add sampled image.")
        .build().expect("Could not build the image sampler set.")
    )
  }
}
