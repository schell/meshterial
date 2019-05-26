use vulkano::buffer::BufferUsage;
use vulkano::buffer::device_local::DeviceLocalBuffer;
use vulkano::descriptor::descriptor_set::{PersistentDescriptorSet, DescriptorSet};
use vulkano::device::Device;
use vulkano::instance::PhysicalDevice;
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::framebuffer::{RenderPassAbstract, Subpass};

use nalgebra::*;

use std::sync::Arc;

pub mod vs {
  vulkano_shaders::shader!{
    ty: "vertex",
    path: "src/shaders/color3d/vert.glsl"
  }
}

pub mod fs {
  vulkano_shaders::shader!{
    ty: "fragment",
    path: "src/shaders/color3d/frag.glsl"
  }
}

#[derive(Debug, Clone)]
pub struct VertexColor3 {
  pub position: [f32; 3],
  pub color: [f32; 4],
}
impl_vertex!(VertexColor3, position, color);


pub struct Color3DPipeline {
  pub pipeline: Arc<GraphicsPipelineAbstract + Send + Sync>,
  pub proj_uniform_device_buffer:Arc<DeviceLocalBuffer<Matrix4<f32>>>,
  pub proj_desc_set: Arc<DescriptorSet + Send + Sync>,
}


impl Color3DPipeline {
  pub fn new(
    render_pass: Arc<RenderPassAbstract + Send + Sync>,
    device: Arc<Device>,
    physical: PhysicalDevice
  ) -> Color3DPipeline {
    let vs = vs::Shader::load(device.clone()).expect("failed to create shader module");
    let fs = fs::Shader::load(device.clone()).expect("failed to create shader module");
    let pipeline = Arc::new(
      GraphicsPipeline::start()
      // We need to indicate the layout of the vertices.
      // The type `SingleBufferDefinition` actually contains a template parameter corresponding
      // to the type of each vertex. But in this code it is automatically inferred.
        .vertex_input_single_buffer::<VertexColor3>()
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
        .cull_mode_back()
      // We have to indicate which subpass of which render pass this pipeline is going to be used
      // in. The pipeline will only be usable from this particular subpass.
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
      // Now that our builder is filled, we call `build()` to obtain an actual pipeline.
        .build(device.clone())
        .unwrap()
    );

    // Uniform stuff
    let proj_uniform_device_buffer:Arc<DeviceLocalBuffer<_>> = DeviceLocalBuffer::new(
      device,
      BufferUsage::uniform_buffer_transfer_destination(),
      physical.queue_families()
    ).expect("Could not create uniform_device_buffer.");

    let proj_desc_set = Arc::new(
      PersistentDescriptorSet::start(pipeline.clone(), 0)
        .add_buffer(proj_uniform_device_buffer.clone()).expect("Could not add uniform_device_buffer.")
        .build()
        .expect("Could not build uniform_desc_set.")
    );

    Color3DPipeline {
      pipeline,
      proj_uniform_device_buffer,
      proj_desc_set,
    }
  }
}
