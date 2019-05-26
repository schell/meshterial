use std::sync::Arc;
use vulkano::buffer::device_local::DeviceLocalBuffer;
use vulkano::descriptor::descriptor_set::DescriptorSet;


pub struct UniformDeviceAndDescriptor<T> {
  pub device_buffer: Arc<DeviceLocalBuffer<T>>,
  pub desc_set: Arc<DescriptorSet + Send + Sync>
}
