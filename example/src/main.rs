// The `vulkano` crate is the main crate that you must use to use Vulkan.
extern crate vulkano;
extern crate vulkano_shaders;
extern crate sdl2;
extern crate nalgebra;
extern crate nalgebra_glm;
extern crate image;
extern crate meshterial;
//extern crate collada;

use collada::document::ColladaDocument;
use collada::PrimitiveElement;
use vulkano::instance::PhysicalDevice;
use vulkano::buffer::{CpuAccessibleBuffer, BufferUsage};
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use nalgebra::Matrix4;
use nalgebra_glm as glm;
use std::time::{Duration, Instant};
use std::path::Path;
use std::collections::HashMap;
//use xml;

use meshterial::*;
use meshterial::utils::*;
//use renderer::pipelines::color3d::*;
use meshterial::pipelines::phong::*;
//use renderer::pipelines::texture2d::*;

fn main() {
  let mut vkr = VkRenderer::new();
  vkr
    .window.set_title("Vulkan Renderer Demo")
    .expect("Could not set window title.");

  // Before we draw we have to create what is called a pipeline. This is similar to an OpenGL
  // program, but much more specific.
  // This pipeline is the colored pipeline, used for drawing colored geometry.
  let instance = vkr.instance.clone();
  let physical = PhysicalDevice::enumerate(&instance)
    .next()
    .expect("no physical device available");

  let doc = ColladaDocument::from_path(Path::new("assets/test.dae"))
    .expect("Could not load collada file.");

  let eff_lib = doc.get_effect_library();

  let mats_to_effs = doc.get_material_to_effect();

  let mut material_vertex_buffers:HashMap<String, Vec<VertexPhong>> = HashMap::new();

  println!("Materials in library: {:#?}", eff_lib.keys());

  if let Some(obj_set) = doc.get_obj_set() {
    obj_set
      .objects
      .iter()
      .for_each(|obj| {
        println!("Object name: {}", obj.name);
        obj
          .geometry
          .iter()
          .for_each(|geom| {
            geom
              .mesh
              .iter()
              .for_each(|prim| {
                match prim {
                  PrimitiveElement::Polylist(_) => {}
                  PrimitiveElement::Triangles(triangles) => {
                    let material = triangles
                      .material
                      .as_ref()
                      .expect("No material!")
                      .clone();
                    let eff = mats_to_effs.get(&material)
                      .expect("Could not find material effect");

                    if !material_vertex_buffers.contains_key(eff) {
                      // Make a new entry!
                      material_vertex_buffers.insert(eff.clone(), vec![]);
                    }

                    let buffer = material_vertex_buffers.get_mut(eff)
                      .expect("This should never happen.");

                    triangles
                      .vertices
                      .iter()
                      .for_each(|(a, b, c)| {
                        // Add the values pointed to by the indices
                        // into the vertex buffer
                        for (vndx, _, may_nndx) in [a, b, c].iter() {
                          let nndx = may_nndx
                            .expect("vertex is missing a normal");
                          let p = obj
                            .vertices
                            .get(*vndx)
                            .expect(&format!("could not get vertex at ndx {}", vndx));
                          let n = obj
                            .normals
                            .get(nndx)
                            .expect(&format!("could not get vertex at ndx {}", vndx));
                          buffer.push(VertexPhong{
                            position: [p.x as f32, p.y as f32, p.z as f32],
                            normal: [n.x as f32, n.y as f32, n.z as f32]
                          });
                        }
                      });
                  }
                }
              })
          });
      });
  }

  let mut material_buffers = vec![];
  for (eff, vertices) in material_vertex_buffers {
    let buffer = CpuAccessibleBuffer::from_iter(vkr.device.clone(), BufferUsage::all(), vertices.iter().cloned())
      .expect("Could not create material vertex buffer.");
    material_buffers.push((eff, buffer));
  };

  // Create the pipeline
  let phong_pipeline = PhongPipeline::new(
    vkr.render_pass.clone(),
    vkr.device.clone(),
    physical.clone(),
    eff_lib.keys().cloned().collect()
  );

  // Set the material uniforms on the pipeline.
  for (name, tech) in eff_lib.iter() {
    let uniform = phong_pipeline
      .materials
      .get(name)
      .expect(&format!("Could not get material {}", name));

    let material = Material {
      emission: tech.emission,
      ambient: tech.ambient,
      diffuse: tech.diffuse,
      specular: tech.specular,
      shininess: tech.shininess
    };

    let src_uniform = phong_pipeline
      .material_buffer_pool
      .next(material)
      .expect("Could not load material into cpu buffer");

    let dest_uniform = uniform.device_buffer.clone();

    vkr.with_command_builder(|cmds| {
      cmds
        .copy_buffer(src_uniform.clone(), dest_uniform)
        .expect("Could not copy material uniform to device buffer.")
    });
  }

  // Set the light uniform on the pipeline.
  {
    let light = Light {
      position: [0.0, 100.0, 0.0],
      _dummy0: [0, 0, 0, 0],
      intensity: [1.0, 1.0, 1.0],
    };
    let src = phong_pipeline
      .light_buffer_pool
      .next(light)
      .expect("Could not load light into cpu buffer.");
    let dest = phong_pipeline.light.device_buffer.clone();
    vkr.with_command_builder(|cmds| {
      cmds
        .copy_buffer(src.clone(), dest)
        .expect("Could not copy light uniform to device.")
    });
  }

  let mut fps = FPSCounter::new();
  let print_at = Duration::from_secs(2);
  let mut last_instant = Instant::now();

  // Initialization is finally finished!

  // In the loop below we are going to submit commands to the GPU. Submitting a command produces
  // an object that implements the `GpuFuture` trait, which holds the resources for as long as
  // they are in use by the GPU.
  let mut event_pump = vkr
    .ctx
    .event_pump()
    .expect("Could not get event_pump()");

  'mainloop: loop {
    if let Some(resized) = vkr.start_next_frame() {
      // Do any updates associated with resizing the window.
      if resized {
        // Update the projection uniform.
        // Set up our uniform buffer.
        //let projection2d:Matrix4<f32> = Matrix4::new_orthographic(
        //  0.0,
        //  vkr.dimensions[0] as f32,
        //  0.0,
        //  vkr.dimensions[1] as f32,
        //  -1.0,
        //  1.0
        //);

        let aspect = vkr.dimensions[0] as f32 / vkr.dimensions[1] as f32;
        let projection3d:Matrix4<f32> = Matrix4::new_perspective(
          aspect,
          std::f32::consts::FRAC_PI_2,
          0.01,
          100.0
        );

        //let src_buffer2d = vkr
        //  .proj_buffer_pool
        //  .next(projection2d)
        //  .expect("Could not load projection2d into cpu buffer");

        let src_buffer3d = vkr
          .proj_buffer_pool
          .next(projection3d)
          .expect("Could not load projection3d into cpu buffer");

        let phong_dest_buffer = phong_pipeline.proj.device_buffer.clone();

        vkr.with_command_builder(|cmds| {
          cmds
            .copy_buffer(src_buffer3d.clone(), phong_dest_buffer)
            .expect("Could not copy color3d projection uniform cpu buffer to the gpu.")
        });
        // In the future we may also want to do this:
        // https://github.com/vulkano-rs/vulkano-examples/blob/master/src/bin/teapot/main.rs#L265
      }

      vkr.begin_rendering();
      // We are now inside the first subpass of the render pass. We can submit
      // draw commands.

      let model:Matrix4<f32> = Matrix4::identity();
      let view:Matrix4<f32> = glm::look_at(
        &glm::vec3(3.0, 3.0, 4.0),
        &glm::vec3(0.0, 0.0, 0.0),
        &glm::vec3(0.0, 1.0, 0.0)
      );
      let modelview = model * view;
      let normal = modelview
        .pseudo_inverse(1e-10)
        .transpose();

      let mut modelviewnormal = meshterial::pipelines::phong::vs::ty::ModelViewNormal {
        model: model.into(),
        view: view.into(),
        normal: normal.into(),
      };
      let mut model:Matrix4<f32> = Matrix4::identity();

      let dynamic_state = vkr.dynamic_state.clone();
      for (eff, buffer) in &material_buffers {
        let eff:&String = eff;
        let material_set = &phong_pipeline
          .materials
          .get(eff)
          .expect("Could not find effect in pipeline materials list.")
          .desc_set;
        vkr.with_command_builder(|cmds| {
          cmds
            .draw(
              phong_pipeline.pipeline.clone(),
              &dynamic_state,
              vec!(buffer.clone()),
              (phong_pipeline.proj.desc_set.clone(), material_set.clone(), phong_pipeline.light.desc_set.clone()),
              modelviewnormal
            ).expect("Could not draw 3d geometry in the main loop.")
        });
      }
      vkr.commit_rendering();
    }

    // Handling the window events in order to close the program when the user wants to close
    // it.
    for event in event_pump.poll_iter() {
      match event {
        Event::Quit {..} | Event::KeyDown { keycode: Some(Keycode::Escape), .. } => {
          break 'mainloop;
        },
        _ => {}
      }
    }

    fps.next_frame();
    let this_instant = Instant::now();
    if this_instant.duration_since(last_instant) >= print_at {
      last_instant = this_instant;
      println!("FPS:{:.2}", fps.current_fps());
    }
  }
}
