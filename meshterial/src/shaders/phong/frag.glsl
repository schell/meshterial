#version 450

layout(set = 1, binding = 0) uniform Material {
  vec4 emission;
  vec4 ambient;
  vec4 diffuse;
  vec4 specular;
  float shininess;
  //float index_of_refraction;
} material;

layout(set = 2, binding = 0) uniform Light {
  vec3 position;
  vec3 intensity;
} light;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

layout(location = 0) out vec4 frag_color;

void main() {
  vec3 n = normalize( normal );
  vec3 s = normalize( light.position - position );
  vec3 v = normalize( -position );
  vec3 r = reflect(-s, n);
  vec4 c = material.ambient +
    material.diffuse * max(dot(s, n), 0.0) +
    material.specular * pow(max(dot(r, v), 0.0), material.shininess);
  frag_color = vec4(light.intensity, 1.0) * c;
  //frag_color = vec4(n.rgb, 1.0);
}
