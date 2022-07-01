mod instrs;
mod stack;

#[cfg(test)]
mod tests;

pub use self::stack::Stack;
use self::{instrs::execute_frame, stack::StackFrameRef};
use super::{super::ExecRegisterSlice, EngineInner};
use crate::{
    engine::{CallParams, CallResults, DedupFuncType, ExecProviderSlice},
    func::{FuncEntityInternal, HostFuncEntity, WasmFuncEntity},
    AsContext,
    AsContextMut,
    Func,
};
use core::cmp;
use wasmi_core::Trap;

/// The possible outcomes of a function execution.
#[derive(Debug, Copy, Clone)]
enum CallOutcome {
    /// Returns the result of the function execution.
    Return {
        /// The returned result values.
        returned: ExecProviderSlice,
    },
    /// Persons a nested function call.
    Call {
        /// The results of the function call.
        results: ExecRegisterSlice,
        /// The called function.
        callee: Func,
        /// The parameters of the function call.
        params: ExecProviderSlice,
    },
}

impl EngineInner {
    /// Executes the given [`Func`] using the given arguments `args` and stores the result into `results`.
    ///
    /// # Errors
    ///
    /// - If the given arguments `args` do not match the expected parameters of `func`.
    /// - If the given `results` do not match the the length of the expected results of `func`.
    /// - When encountering a Wasm trap during the execution of `func`.
    pub fn execute_func<Params, Results>(
        &mut self,
        mut ctx: impl AsContextMut,
        func: Func,
        params: Params,
        results: Results,
    ) -> Result<<Results as CallResults>::Results, Trap>
    where
        Params: CallParams,
        Results: CallResults,
    {
        match func.as_internal(&ctx) {
            FuncEntityInternal::Wasm(wasm_func) => {
                let signature = wasm_func.signature();
                let frame = self.initialize_args(wasm_func, params);
                let returned_values = self.execute_frame(&mut ctx, frame)?;
                let results = self.return_result(signature, returned_values, results);
                Ok(results)
            }
            FuncEntityInternal::Host(_host_func) => {
                todo!()
            }
        }
    }

    /// Initializes the registers with the given arguments `params`.
    ///
    /// # Note
    ///
    /// This initializes the registers holding the parameters of the called
    /// root function.
    /// Registers for the local variables are initialized to zero.
    fn initialize_args(&mut self, func: &WasmFuncEntity, params: impl CallParams) -> StackFrameRef {
        self.stack.init(func, params)
    }

    /// Executes the given Wasm [`Func`] using the given arguments `args` and stores the result into `results`.
    ///
    /// # Note
    ///
    /// The caller is required to ensure that the given `func` actually is a Wasm function.
    ///
    /// # Errors
    ///
    /// - If the given arguments `args` do not match the expected parameters of `func`.
    /// - If the given `results` do not match the the length of the expected results of `func`.
    /// - When encountering a Wasm trap during the execution of `func`.
    fn execute_frame(
        &mut self,
        mut ctx: impl AsContextMut,
        mut frame: StackFrameRef,
    ) -> Result<ExecProviderSlice, Trap> {
        'outer: loop {
            let mut view = self.stack.frame_at(frame);
            match execute_frame(&mut ctx, &self.code_map, &self.res, &mut view)? {
                CallOutcome::Return { returned } => {
                    // Pop the last frame from the function frame stack and
                    // continue executing it OR finish execution if the call
                    // stack is empty.
                    match self.stack.pop_frame(returned, &self.res) {
                        Some(next_frame) => {
                            frame = next_frame;
                            continue 'outer;
                        }
                        None => {
                            // We just tried to pop the root stack frame.
                            // Therefore we need to return since the execution
                            // is over at this point.
                            return Ok(returned);
                        }
                    }
                }
                CallOutcome::Call {
                    results,
                    callee,
                    params,
                } => {
                    match callee.as_internal(&ctx) {
                        FuncEntityInternal::Wasm(wasm_func) => {
                            frame = self.stack.push_frame(wasm_func, results, params, &self.res);
                        }
                        FuncEntityInternal::Host(host_func) => {
                            let host_func = host_func.clone();
                            self.execute_host_func(&mut ctx, frame, results, host_func, params)?;
                        }
                    };
                }
            }
        }
    }

    /// Executes the given host function.
    ///
    /// # Errors
    ///
    /// - If the host function returns a host side error or trap.
    #[inline(never)]
    fn execute_host_func<C>(
        &mut self,
        _ctx: C,
        _caller: StackFrameRef,
        _results: ExecRegisterSlice,
        host_func: HostFuncEntity<<C as AsContext>::UserState>,
        _params: ExecProviderSlice,
    ) -> Result<(), Trap>
    where
        C: AsContextMut,
    {
        // The host function signature is required for properly
        // adjusting, inspecting and manipulating the value stack.
        let (input_types, output_types) = self
            .res
            .func_types
            .resolve_func_type(host_func.signature())
            .params_results();
        // In case the host function returns more values than it takes
        // we are required to extend the value stack.
        let len_inputs = input_types.len();
        let len_outputs = output_types.len();
        let _max_inout = cmp::max(len_inputs, len_outputs);
        // self.value_stack.reserve(max_inout)?;
        // if len_outputs > len_inputs {
        //     let delta = len_outputs - len_inputs;
        //     self.value_stack.extend_zeros(delta)?;
        // }
        // let params_results = FuncParams::new(
        //     self.value_stack.peek_as_slice_mut(max_inout),
        //     len_inputs,
        //     len_outputs,
        // );
        // // Now we are ready to perform the host function call.
        // // Note: We need to clone the host function due to some borrowing issues.
        // //       This should not be a big deal since host functions usually are cheap to clone.
        // host_func.call(ctx.as_context_mut(), instance, params_results)?;
        // // If the host functions returns fewer results than it receives parameters
        // // the value stack needs to be shrinked for the delta.
        // if len_outputs < len_inputs {
        //     let delta = len_inputs - len_outputs;
        //     self.value_stack.drop(delta);
        // }
        // // At this point the host function has been called and has directly
        // // written its results into the value stack so that the last entries
        // // in the value stack are the result values of the host function call.
        // Ok(())
        todo!()
    }

    /// Writes the results of the function execution back into the `results` buffer.
    ///
    /// # Panics
    ///
    /// - If the `results` buffer length does not match the remaining amount of stack values.
    fn return_result<Results>(
        &mut self,
        func_type: DedupFuncType,
        returned_values: ExecProviderSlice,
        results: Results,
    ) -> <Results as CallResults>::Results
    where
        Results: CallResults,
    {
        let result_types = self.res.func_types.resolve_func_type(func_type).results();
        let returned_values = self.res.provider_slices.resolve(returned_values);
        assert_eq!(
            returned_values.len(),
            results.len_results(),
            "expected {} values on the stack after function execution but found {}",
            results.len_results(),
            returned_values.len(),
        );
        assert_eq!(results.len_results(), result_types.len());
        let resolve_cref = |cref| {
            self.res
                .const_pool
                .resolve(cref)
                .unwrap_or_else(|| panic!("failed to resolve constant reference: {:?}", cref))
        };
        self.stack
            .finalize(result_types, resolve_cref, returned_values, results)
    }
}
