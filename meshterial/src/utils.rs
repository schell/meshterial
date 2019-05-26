use std::time::{Instant, Duration};


pub struct DurationMeasurement {
  start: Instant,
}


impl DurationMeasurement {
  pub fn starting_now() -> DurationMeasurement {
    DurationMeasurement {
      start: Instant::now()
    }
  }


  pub fn duration_since_start(&self) -> Duration {
     Instant::now().duration_since(self.start)
  }
}


pub fn measure<T, F: FnOnce() -> T> (f: F) -> (T, Duration) {
  let m = DurationMeasurement::starting_now();
  let t = f();
  (t, m.duration_since_start())
}


pub struct FPSCounter {
  buffer: [f32; 600],
  index: usize,
  last_instant: Instant
}


impl FPSCounter {
  pub fn new() -> FPSCounter {
    FPSCounter {
      buffer: [0.0; 600],
      index: 0,
      last_instant: Instant::now(),
    }
  }
  pub fn next_frame(&mut self) -> f32 {
    let this_instant = Instant::now();
    let delta = this_instant.duration_since(self.last_instant);
    let dt = delta.as_secs() as f32 + (delta.subsec_nanos() as f32 / 1_000_000_000.0);
    self.last_instant = this_instant;
    self.buffer[self.index] = dt;
    self.index = (self.index + 1) % self.buffer.len();
    dt
  }

  pub fn avg_frame_delta(&self) -> f32 {
    self.buffer.iter().fold(0.0, |sum, dt| sum + dt) / self.buffer.len() as f32
  }

  pub fn current_fps(&self) -> f32 {
    1.0 / self.avg_frame_delta()
  }

  pub fn last_delta(&self) -> f32 {
    self.buffer[self.index]
  }

  pub fn frames(&self) -> &[f32; 600] {
    &self.buffer
  }
}

impl Default for FPSCounter {
  fn default() -> FPSCounter {
    FPSCounter::new()
  }
}
