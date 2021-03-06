use std::convert::TryInto;
use std::ffi::{CStr, CString};
use tensorrt_sys::{
    context_get_name, context_set_name, destroy_excecution_context, execute, execute_v2, execute_bindings_v2, Context_t,
    set_optimization_profile, set_binding_shape_dims2
};

pub struct Context<'a> {
    pub(crate) internal_context: *mut Context_t,
    pub(crate) _engine: &'a crate::engine::Engine,
}

impl<'a> Context<'a> {
    pub fn set_name(&mut self, context_name: &str) {
        unsafe {
            context_set_name(
                self.internal_context,
                CString::new(context_name).unwrap().as_ptr(),
            )
        };
    }

    pub fn get_name(&self) -> String {
        let context_name = unsafe {
            let raw_context_name = context_get_name(self.internal_context);
            CStr::from_ptr(raw_context_name)
        };
        context_name.to_str().unwrap().to_string()
    }

    pub fn execute(
        &self,
        input_data: &[f32],
        input_binding_index: i32,
        output_data: &mut[f32],
        output_data_size: usize,
        ouptut_binding_index: i32,
    ) {
        unsafe {
            execute(
                self.internal_context,
                input_data.as_ptr(),
                input_data.len() * std::mem::size_of::<f32>(),
                input_binding_index.try_into().unwrap(),
                output_data.as_mut_ptr(),
                output_data_size * std::mem::size_of::<f32>(),
                ouptut_binding_index.try_into().unwrap(),
            );
        }
    }

    pub fn execute_v2(
        &self,
        input_data: &[f32],
        input_binding_index: i32,
        output_data: &mut[f32],
        output_data_size: usize,
        ouptut_binding_index: i32,
    ) {
        unsafe {
            execute_v2(
                self.internal_context,
                input_data.as_ptr(),
                input_data.len() * std::mem::size_of::<f32>(),
                input_binding_index.try_into().unwrap(),
                output_data.as_mut_ptr(),
                output_data_size * std::mem::size_of::<f32>(),
                ouptut_binding_index.try_into().unwrap(),
            );
        }
    }

    pub fn execute_bindings_v2(
        &self,
        bindings: *const std::ffi::c_void
    ) {
        unsafe {
            execute_bindings_v2(
                self.internal_context,
                bindings
            );
        }
    }

    pub fn set_optimization_profile(&self, level: i32) {
        unsafe {
            set_optimization_profile(self.internal_context, level);
        }
    }

    pub fn set_binding_shape_dims2(&self, index: i32, d0: i32, d1: i32) {
        unsafe {
            set_binding_shape_dims2(self.internal_context, index, d0, d1);
        }
    }
}

impl<'a> Drop for Context<'a> {
    fn drop(&mut self) {
        unsafe { destroy_excecution_context(self.internal_context) };
    }
}

#[cfg(test)]
mod tests {
    // use super::*;
    // use crate::engine::Engine;
    // use crate::runtime::{Logger, Runtime};
    // use std::fs::File;
    // use std::io::prelude::*;

    // fn setup_engine_test() -> Engine {
    //     let logger = Logger::new();
    //     let runtime = Runtime::new(&logger);
    //
    //     let mut f = File::open("../tensorrt-sys/resnet34-unet-Aug25-07-25-16-best.engine").unwrap();
    //     let mut buffer = Vec::new();
    //     f.read_to_end(&mut buffer).unwrap();
    //
    //     Engine::new(runtime, buffer)
    // }
    //
    // #[test]
    // fn set_context_name() {
    //     let engine = setup_engine_test();
    //     let mut context = engine.create_execution_context();
    //
    //     context.set_name("Mason");
    //     let name = context.get_name();
    //
    //     assert_eq!(name, "Mason".to_string());
    // }
}
