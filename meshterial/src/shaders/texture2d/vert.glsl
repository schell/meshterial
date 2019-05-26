#version 450

layout(set = 0, binding = 0) uniform Projection {
  mat4 mat;
} proj;

layout(push_constant) uniform Matrices {
  mat4 model;
  mat4 view;
} mats;

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 uv;

layout(location = 0) out vec2 out_uv;

void main() {
  out_uv = uv;
  gl_Position = proj.mat * mats.model * mats.view * vec4(position, 0.0, 1.0);
}
