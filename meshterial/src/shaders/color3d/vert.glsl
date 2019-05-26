#version 450

layout(set = 0, binding = 0) uniform Projection {
  mat4 mat;
} proj;

layout(push_constant) uniform ModelView {
  mat4 model;
  mat4 view;
} modelview;

layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;

layout(location = 0) out vec4 out_color;

void main() {
  out_color = color;
  gl_Position = proj.mat * modelview.view * modelview.model * vec4(position, 1.0);
}
