use vulkano::buffer::BufferUsage;
use vulkano::buffer::cpu_pool::CpuBufferPool;
use vulkano::buffer::device_local::DeviceLocalBuffer;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::Device;
use vulkano::instance::PhysicalDevice;
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::framebuffer::{RenderPassAbstract, Subpass};
use nalgebra::*;
use std::sync::Arc;
use std::collections::HashMap;

pub use super::uniform::*;

pub mod vs {
  vulkano_shaders::shader!{
    ty: "vertex",
    path: "src/shaders/phong/vert.glsl"
  }
}

pub mod fs {
  vulkano_shaders::shader!{
    ty: "fragment",
    path: "src/shaders/phong/frag.glsl"
  }
}

pub use self::fs::ty::{Material, Light};

#[derive(Debug, Clone)]
pub struct VertexPhong {
  pub position: [f32; 3],
  pub normal: [f32; 3]
}
impl_vertex!(VertexPhong, position, normal);


pub struct PhongPipeline {
  pub pipeline: Arc<GraphicsPipelineAbstract + Send + Sync>,
  pub proj: UniformDeviceAndDescriptor<Matrix4<f32>>,
  pub light: UniformDeviceAndDescriptor<Light>,
  pub light_buffer_pool: CpuBufferPool<Light>,
  pub materials: HashMap<String, UniformDeviceAndDescriptor<Material>>,
  pub material_buffer_pool: CpuBufferPool<Material>,
}


impl PhongPipeline {
  /// Creates a new PhongPipeline.
  pub fn new(
    render_pass: Arc<RenderPassAbstract + Send + Sync>,
    device: Arc<Device>,
    physical: PhysicalDevice,
    material_names: Vec<String>
  ) -> PhongPipeline {
    let vs = vs::Shader::load(device.clone()).expect("failed to create shader module");
    let fs = fs::Shader::load(device.clone()).expect("failed to create shader module");
    let pipeline = Arc::new(
      GraphicsPipeline::start()
      // We need to indicate the layout of the vertices.
      // The type `SingleBufferDefinition` actually contains a template parameter corresponding
      // to the type of each vertex. But in this code it is automatically inferred.
        .vertex_input_single_buffer::<VertexPhong>()
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
        .depth_stencil_simple_depth()
      // Does it blend?
        .blend_alpha_blending()
      // Culling
        //.cull_mode_back()
      // We have to indicate which subpass of which render pass this pipeline is going to be used
      // in. The pipeline will only be usable from this particular subpass.
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
      // Now that our builder is filled, we call `build()` to obtain an actual pipeline.
        .build(device.clone())
        .unwrap()
    );

    // Uniform stuff
    let proj = {
      let device_buffer:Arc<DeviceLocalBuffer<_>> = DeviceLocalBuffer::new(
        device.clone(),
        BufferUsage::uniform_buffer_transfer_destination(),
        physical.queue_families()
      ).expect("Could not create uniform_device_buffer.");

      let desc_set = Arc::new(
        PersistentDescriptorSet::start(pipeline.clone(), 0)
          .add_buffer(device_buffer.clone()).expect("Could not add uniform_device_buffer.")
          .build()
          .expect("Could not build uniform_desc_set.")
      );

      UniformDeviceAndDescriptor {
        device_buffer, desc_set
      }
    };

    let mut materials = HashMap::new();
    for name in material_names {
      let uniform = {
        let device_buffer:Arc<DeviceLocalBuffer<_>> = DeviceLocalBuffer::new(
          device.clone(),
          BufferUsage::uniform_buffer_transfer_destination(),
          physical.queue_families()
        ).expect("Could not create uniform_device_buffer.");

        let desc_set = Arc::new(
          PersistentDescriptorSet::start(pipeline.clone(), 1)
            .add_buffer(device_buffer.clone()).expect("Could not add uniform_device_buffer.")
            .build()
            .expect("Could not build uniform_desc_set.")
        );

        UniformDeviceAndDescriptor {
          device_buffer, desc_set
        }
      };

      materials.insert(name, uniform);
    }

    let light = {
      let device_buffer:Arc<DeviceLocalBuffer<_>> = DeviceLocalBuffer::new(
        device.clone(),
        BufferUsage::uniform_buffer_transfer_destination(),
        physical.queue_families()
      ).expect("Could not create uniform_device_buffer.");

      let desc_set = Arc::new(
        PersistentDescriptorSet::start(pipeline.clone(), 2)
          .add_buffer(device_buffer.clone()).expect("Could not add uniform_device_buffer.")
          .build()
          .expect("Could not build uniform_desc_set.")
      );

      UniformDeviceAndDescriptor {
        device_buffer, desc_set
      }
    };


    let material_buffer_pool = CpuBufferPool::upload(device.clone());
    let light_buffer_pool = CpuBufferPool::upload(device.clone());

    PhongPipeline {
      pipeline,
      proj,
      materials,
      material_buffer_pool,
      light,
      light_buffer_pool
    }
  }


}
