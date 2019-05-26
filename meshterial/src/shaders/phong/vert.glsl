#version 450

layout(set = 0, binding = 0) uniform Projection {
  mat4 mat;
} proj;

layout(push_constant) uniform ModelViewNormal {
  mat4 model;
  mat4 view;
  mat4 normal;
} mats;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

layout(location = 0) out vec3 out_position;
layout(location = 1) out vec3 out_normal;

void main() {
  mat4 modelview = mats.model * mats.view;
  vec4 pos = modelview * vec4(position, 1.0);
  out_normal = normalize( mats.normal * vec4(normal, 1.0) ).xyz;
  out_position = pos.xyz;
  gl_Position = proj.mat * modelview * pos;
}
