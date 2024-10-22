pub use self::call::{dispatch_host_func, ResumableHostError};
use super::{cache::CachedInstance, InstructionPtr, Stack};
use crate::{
    core::{hint, TrapCode, UntypedVal},
    engine::{
        bytecode::{index, BlockFuel, Const16, Instruction, Reg},
        code_map::CodeMap,
        executor::stack::{CallFrame, FrameRegisters, ValueStack},
        utils::unreachable_unchecked,
        DedupFuncType, EngineFunc,
    },
    ir::ShiftAmount,
    memory::DataSegment,
    store::StoreInner,
    table::ElementSegment,
    Error, Func, FuncRef, Global, Memory, Store, Table,
};
use std::{vec, vec::Vec};


#[cfg(doc)]
use crate::Instance;

mod binary;
mod branch;
mod call;
mod comparison;
mod conversion;
mod copy;
mod global;
mod load;
mod memory;
mod return_;
mod select;
mod store;
mod table;
mod unary;

macro_rules! forward_return {
    ($expr:expr) => {{
        if hint::unlikely($expr.is_break()) {
            return Ok(());
        }
    }};
}

/// Executes compiled function instructions until execution returns from the root function.
///
/// # Errors
///
/// If the execution encounters a trap.
#[inline(never)]
pub fn execute_instrs<'engine, T>(
    store: &mut Store<T>,
    stack: &'engine mut Stack,
    code_map: &'engine CodeMap,
) -> Result<(), Error> {
    let instance = stack.calls.instance_expect();
    let cache = CachedInstance::new(&mut store.inner, instance);
    Executor::new(stack, code_map, cache).execute(store)
}

/// An execution context for executing a Wasmi function frame.
#[derive(Debug)]
struct Executor<'engine> {
    /// Stores the value stack of live values on the Wasm stack.
    sp: FrameRegisters,
    /// The pointer to the currently executed instruction.
    ip: InstructionPtr,
    /// The cached instance and instance related data.
    cache: CachedInstance,
    /// The value and call stacks.
    stack: &'engine mut Stack,
    /// The static resources of an [`Engine`].
    ///
    /// [`Engine`]: crate::Engine
    code_map: &'engine CodeMap,
}

impl<'engine> Executor<'engine> {
    /// Creates a new [`Executor`] for executing a Wasmi function frame.
    #[inline(always)]
    pub fn new(
        stack: &'engine mut Stack,
        code_map: &'engine CodeMap,
        cache: CachedInstance,
    ) -> Self {
        let frame = stack
            .calls
            .peek()
            .expect("must have call frame on the call stack");
        // Safety: We are using the frame's own base offset as input because it is
        //         guaranteed by the Wasm validation and translation phase to be
        //         valid for all register indices used by the associated function body.
        let sp = unsafe { stack.values.stack_ptr_at(frame.base_offset()) };
        let ip = frame.instr_ptr();
        Self {
            sp,
            ip,
            cache,
            stack,
            code_map,
        }
    }

    /// Executes the function frame until it returns or traps.
    #[inline(always)]
    fn execute<T>(mut self, store: &mut Store<T>) -> Result<(), Error> {
        use Instruction as Instr;
        loop {
            let instr = *self.ip.get();
            // update the runtime signature with the current instruction
            // we map the instruction to a unique 64-bit prime number
            let instr_prime = match instr.clone() {
                Instr::Const32 { .. } => 0xe3a461c24c1edf67,
                Instr::I64Const32 { .. } => 0x93e0632ef59fbf8d,
                Instr::F64Const32 { .. } => 0xcf96777f6bf48827,
                Instr::Register { .. } => 0xa1a9bcb9fec5fdfb,
                Instr::Register2 { .. } => 0xbee08b06e6ab17f5,
                Instr::Register3 { .. } => 0xb448b4a7d84f751f,
                Instr::RegisterList { .. } => 0xb918e0472d8c224f,
                Instr::CallIndirectParams { .. } => 0xbf382b4acfe7644b,
                Instr::CallIndirectParamsImm16 { .. } => 0xd853e6a184c25f0d,
                Instr::Trap { .. } => 0xb18d650b9f5998a7,
                Instr::ConsumeFuel { .. } => 0xe6118441cda42713,
                Instr::Return => 0xc8b8b1c1bcbd90e5,
                Instr::ReturnReg { .. } => 0xbaab8e9341e08dbf,
                Instr::ReturnReg2 { .. } => 0xa73d1157b48ca275,
                Instr::ReturnReg3 { .. } => 0xa6ffa6328ec1eb81,
                Instr::ReturnImm32 { .. } => 0x83307e10c33705a3,
                Instr::ReturnI64Imm32 { .. } => 0xe8a6034d312d2135,
                Instr::ReturnF64Imm32 { .. } => 0xdc28177292727dc9,
                Instr::ReturnSpan { .. } => 0xda75d553c9a0933b,
                Instr::ReturnMany { .. } => 0xa5084225f2090f95,
                Instr::ReturnNez { .. } => 0xe31a3fb6dd7f310d,
                Instr::ReturnNezReg { .. } => 0xbfd53817fb0381e7,
                Instr::ReturnNezReg2 { .. } => 0xd85809e11f54d745,
                Instr::ReturnNezImm32 { .. } => 0xddb81fb1a74a83b1,
                Instr::ReturnNezI64Imm32 { .. } => 0x860254a7c93ec93f,
                Instr::ReturnNezF64Imm32 { .. } => 0xb2ee1c9ad5f914bb,
                Instr::ReturnNezSpan { .. } => 0xec3158e4f69f44df,
                Instr::ReturnNezMany { .. } => 0xc6cdd0d8f17fe649,
                Instr::Branch { .. } => 0xef66bf425478625b,
                Instr::BranchCmpFallback { .. } => 0x87d943ccc553c97f,
                Instr::BranchI32And { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0xf16d67d2a7dbc15b
                }
                Instr::BranchI32AndImm { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0xd97e76e4a08a4169
                }
                Instr::BranchI32Or { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0xac6e6dcc9eb6cbff
                }
                Instr::BranchI32OrImm { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0xa36564ae5f8bcf13
                }
                Instr::BranchI32Xor { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0xa3fb8b494d435729
                }
                Instr::BranchI32XorImm { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0xd8a580b0d15cf0ab
                }
                Instr::BranchI32AndEqz { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0xc118754f6fd4adc1
                }
                Instr::BranchI32AndEqzImm { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0xa90fbb32f7b47dc7
                }
                Instr::BranchI32OrEqz { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0xa1bf533d0d3f0635
                }
                Instr::BranchI32OrEqzImm { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0xfe99000769fe6ddd
                }
                Instr::BranchI32XorEqz { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0xe2ade8751fc2e9a3
                }
                Instr::BranchI32XorEqzImm { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0xc2c831b19dd7b0d3
                }
                Instr::BranchI32Eq { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0xa9504bf5d4a47f69
                }
                Instr::BranchI32EqImm { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0xcc68c4fcdd5df33b
                }
                Instr::BranchI32Ne { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0xc574d8a05da369d3
                }
                Instr::BranchI32NeImm { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0xcad08b87db831f77
                }
                Instr::BranchI32LtS { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0xc590acad04f1f7b9
                }
                Instr::BranchI32LtSImm { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0xd4d918a2cfb5323d
                }
                Instr::BranchI32LtU { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0xc4999a7e79065d73
                }
                Instr::BranchI32LtUImm { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0xf4fbdab953a405df
                }
                Instr::BranchI32LeS { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0x98a04abe0fa4ce01
                }
                Instr::BranchI32LeSImm { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0xa756dc299bd21ea7
                }
                Instr::BranchI32LeU { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0xebe5a83153067f95
                }
                Instr::BranchI32LeUImm { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0xd6adc84185c3b835
                }
                Instr::BranchI32GtS { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0xc77aef230f5cb5c1
                }
                Instr::BranchI32GtSImm { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0xb288abe58caf78fd
                }
                Instr::BranchI32GtU { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0xdd85783639dea14b
                }
                Instr::BranchI32GtUImm { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0xc95d435e3bd01389
                }
                Instr::BranchI32GeS { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0xe448369b7242bd3b
                }
                Instr::BranchI32GeSImm { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0xd3ed1490c07aec79
                }
                Instr::BranchI32GeU { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0xcbdfc0da7497aca9
                }
                Instr::BranchI32GeUImm { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0xd01255cca5331a55
                }
                Instr::BranchI64Eq { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0xd224c9cfe6c84099
                }
                Instr::BranchI64EqImm { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0xb1f5e1ce9cb796ed
                }
                Instr::BranchI64Ne { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0xa015db66e4480f37
                }
                Instr::BranchI64NeImm { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0xc9534063141f1b6d
                }
                Instr::BranchI64LtS { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0xf3c68e18c0fc1c3b
                }
                Instr::BranchI64LtSImm { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0xfaadf3a5cd945423
                }
                Instr::BranchI64LtU { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0xe12e4e46df02fc2f
                }
                Instr::BranchI64LtUImm { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0xb3476ce898e10f3d
                }
                Instr::BranchI64LeS { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0xfb1cbfc1097a9473
                }
                Instr::BranchI64LeSImm { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0xb4167d6222fadaf7
                }
                Instr::BranchI64LeU { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0xb2932efcea953cab
                }
                Instr::BranchI64LeUImm { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0x821f8f708d1f974f
                }
                Instr::BranchI64GtS { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0xde5463b08e9f4729
                }
                Instr::BranchI64GtSImm { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0xd765407968c91f01
                }
                Instr::BranchI64GtU { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0xe2c63c2c0678900b
                }
                Instr::BranchI64GtUImm { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0xd035ff821066bb9d
                }
                Instr::BranchI64GeS { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0xe49707e335868fa5
                }
                Instr::BranchI64GeSImm { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0xf857874dc48a27e9
                }
                Instr::BranchI64GeU { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0x8b3ce0fa63214359
                }
                Instr::BranchI64GeUImm { lhs, .. } => {
                    self.get_register(lhs).to_bits() ^ 0x93f90f4418d24385
                }
                Instr::BranchF32Eq { lhs, .. } => 
                {
                    self.get_register(lhs).to_bits() ^ 0x8647b33a7b8d4ea9
                }
                Instr::BranchF32Ne { lhs, .. } => 
                {
                    self.get_register(lhs).to_bits() ^ 0x9efcbece1096b201
                }
                Instr::BranchF32Lt { lhs, .. } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xb2ab8327611d4843
                }
                Instr::BranchF32Le { lhs, .. } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xfdb94010ae03ebad 
                }
                Instr::BranchF32Gt { lhs, .. } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xc74489c6752ef2e3
                }
                Instr::BranchF32Ge { lhs, .. } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xb2588add33b6dc8d
                }
                Instr::BranchF64Eq { lhs, .. } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xb0f911188eef530b
                }
                Instr::BranchF64Ne { lhs, .. } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xb3a436328722e3af
                }
                Instr::BranchF64Lt { lhs, .. } => 
                {
                    self.get_register(lhs).to_bits() ^ 0x996ae1e7999d71a5
                }
                Instr::Copy { .. } => 0xf476618f2886dc2f,
                Instr::Copy2 { .. } => 0x81e0ef8904c1cfd5,
                Instr::CopyImm32 { .. } => 0xaafc797a3f40deeb,
                Instr::CopyI64Imm32 { .. } => 0xe6dddd163140692f,
                Instr::CopyF64Imm32 { .. } => 0x976bb2d6ce6f3ccf,
                Instr::CopySpan { .. } => 0x84f01169f85f4fff,
                Instr::CopySpanNonOverlapping { .. } => 0xb32e9b533c4e6e29,
                Instr::CopyMany { .. } => 0xe63c8f65639ebb8f,
                Instr::CopyManyNonOverlapping { .. } => 0xeeb9a195160ac2d7,
                Instr::ReturnCallInternal0 { .. } => 0xec2eb6bd5a5fc313,
                Instr::ReturnCallInternal { .. } => 0xb833133b9b99a663,
                Instr::ReturnCallImported0 { .. } => 0xe14f7c46f5f83b6b,
                Instr::ReturnCallImported { .. } => 0x84e5c91bee77ebb7,
                Instr::ReturnCallIndirect0 { .. } => 0xc5aadca828024d75,
                Instr::ReturnCallIndirect { .. } => 0x814db79997981ca9,
                Instr::CallInternal0 { .. } => 0xfbb357448a8642c3,
                Instr::CallInternal { .. } => 0xab093cfe38b97547,
                Instr::CallImported0 { .. } => 0xe866c937356994c5,
                Instr::CallImported { .. } => 0xa9b3f7092e7cd01b,
                Instr::CallIndirect0 { .. } => 0x89fdcc51af24bead,
                Instr::CallIndirect { .. } => 0xbda3e8601077a917,
                Instr::Select { .. } => 0xcab5aefcb578755f,
                Instr::SelectImm32 { .. } => 0xe640723b1c13c87f,
                Instr::SelectI64Imm32 { .. } => 0xdcdfa8f4a8043ef7,
                Instr::SelectF64Imm32 { .. } => 0x9bbf27a9403e07e3,
                Instr::RefFunc { .. } => 0xd1cd7a96bb99ad23,
                Instr::TableGet { .. } => 0x90f6c6bb3c114319,
                Instr::TableGetImm { .. } => 0x9595b2107e23cb21,
                Instr::TableSize { .. } => 0xd396ced918e61bc5,
                Instr::TableSet { .. } => 0xf5b649b2b404d197,
                Instr::TableSetAt { .. } => 0xcdebf347b50872d3,
                Instr::TableCopy { .. } => 0xf422dd12f6642265,
                Instr::TableCopyTo { .. } => 0xfe2f83b88da7fa03,
                Instr::TableCopyFrom { .. } => 0xfe202c3d504679e1,
                Instr::TableCopyFromTo { .. } => 0x9d9ebedd147ee0c3,
                Instr::TableCopyExact { .. } => 0x9dcc8c066927a9db,
                Instr::TableCopyToExact { .. } => 0xbe7fae07ca7d32ef,
                Instr::TableCopyFromExact { .. } => 0x88d7ecf054f2807d,
                Instr::TableCopyFromToExact { .. } => 0xff641f66fa9a63d3,
                Instr::TableInit { .. } => 0x82bb6ff383050763,
                Instr::TableInitTo { .. } => 0x932d2a71ef983f85,
                Instr::TableInitFrom { .. } => 0xd8cdfe120accedcf,
                Instr::TableInitFromTo { .. } => 0xebca0cc0890416c5,
                Instr::TableInitExact { .. } => 0xc626dd2b280ae6e3,
                Instr::TableInitToExact { .. } => 0xdace86517a593a71,
                Instr::TableInitFromExact { .. } => 0xe5c82cb9a1eac895,
                Instr::TableInitFromToExact { .. } => 0xe5450bc5eb2d7631,
                Instr::TableFill { .. } => 0xe4e6730feefdc50f,
                Instr::TableFillAt { .. } => 0xc23bd8546b888c6b,
                Instr::TableFillExact { .. } => 0x98f8babdc99204f3,
                Instr::TableFillAtExact { .. } => 0x8e49dd000adf0689,
                Instr::TableGrow { .. } => 0x9e61c8c958c6b891,
                Instr::TableGrowImm { .. } => 0x927de647f4278045,
                Instr::ElemDrop { .. } => 0xbc4deb8b398e8a67,
                Instr::DataDrop { .. } => 0xaf73214c7ebdae49,
                Instr::MemorySize { .. } => 0xc99e9ec6fd30df43,
                Instr::MemoryGrow { .. } => 0x902226df112aa763,
                Instr::MemoryGrowBy { .. } => 0xded192652730b3f3,
                Instr::MemoryCopy { .. } => 0xf84118c356d104eb,
                Instr::MemoryCopyTo { .. } => 0xf734ad61d9950a83,
                Instr::MemoryCopyFrom { .. } => 0x9a5467d705c1581d,
                Instr::MemoryCopyFromTo { .. } => 0xc087075594f5ef01,
                Instr::MemoryCopyExact { .. } => 0xa300dfa09a5404db,
                Instr::MemoryCopyToExact { .. } => 0xdecaec4c2f332687,
                Instr::MemoryCopyFromExact { .. } => 0xd973b50e06e7190d,
                Instr::MemoryCopyFromToExact { .. } => 0xd09c8ab6e9e82db3,
                Instr::MemoryFill { .. } => 0x853e90844bc5b9c1,
                Instr::MemoryFillAt { .. } => 0xba3e3f0c5214daf9,
                Instr::MemoryFillImm { .. } => 0xebb50579d7dc30cb,
                Instr::MemoryFillExact { .. } => 0x82bf6b083bb7ab07,
                Instr::MemoryFillAtImm { .. } => 0xb721067ce69b7335,
                Instr::MemoryFillAtExact { .. } => 0xafa4befd04378e05,
                Instr::MemoryFillImmExact { .. } => 0x81e2eca79e7e30c1,
                Instr::MemoryFillAtImmExact { .. } => 0xf4c5f89614b61c1b,
                Instr::MemoryInit { .. } => 0xa30ffa4319eca43b,
                Instr::MemoryInitTo { .. } => 0x8bc41f3bd9e53945,
                Instr::MemoryInitFrom { .. } => 0x922c8e0448c76ad1,
                Instr::MemoryInitFromTo { .. } => 0xdc74c7e86a7d9527,
                Instr::MemoryInitExact { .. } => 0xe8e1aabd32798baf,
                Instr::MemoryInitToExact { .. } => 0xa4612593e32593ab,
                Instr::MemoryInitFromExact { .. } => 0xe6576301838fe52f,
                Instr::MemoryInitFromToExact { .. } => 0xb2b93bf089ba4b27,
                Instr::GlobalGet { .. } => 0x8d923111b80b5901,
                Instr::GlobalSet { .. } => 0xe498f909f87cf3d7,
                Instr::GlobalSetI32Imm16 { .. } => 0xbeceb62a094167cf,
                Instr::GlobalSetI64Imm16 { .. } => 0xb255daab1ca25487,
                Instr::I32Load { .. } => 0xdf5b9b6fa80f3631,
                Instr::I32LoadAt { .. } => 0xf78ad97d27554aab,
                Instr::I32LoadOffset16 { .. } => 0x8d191c3c9f983b7d,
                Instr::I64Load { .. } => 0xcde7973deae4d139,
                Instr::I64LoadAt { .. } => 0xc07cc699947471df,
                Instr::I64LoadOffset16 { .. } => 0xbfd2b00e2b3c39d5,
                Instr::F32Load { .. } => 0xef1fbab218f04407,
                Instr::F32LoadAt { .. } => 0xa8306192cd73002d,
                Instr::F32LoadOffset16 { .. } => 0xed0992f6c6239c7f,
                Instr::F64Load { .. } => 0xf6689ac5b352c02f,
                Instr::F64LoadAt { .. } => 0x97f205959c2a3d0b,
                Instr::F64LoadOffset16 { .. } => 0x94fbb4628a79462b,
                Instr::I32Load8s { .. } => 0xfbb04e5f0a302d7b,
                Instr::I32Load8sAt { .. } => 0x8e95f3bd70e298e7,
                Instr::I32Load8sOffset16 { .. } => 0xb736c7c8935178f5,
                Instr::I32Load8u { .. } => 0xf0e219ca1d327f63,
                Instr::I32Load8uAt { .. } => 0xc5ca3a6dc78a1a5d,
                Instr::I32Load8uOffset16 { .. } => 0xc1932ac6c5cd54ff,
                Instr::I32Load16s { .. } => 0xe74c775c66d1dac7,
                Instr::I32Load16sAt { .. } => 0xbc3c7a6541752f39,
                Instr::I32Load16sOffset16 { .. } => 0x98c1f9f35f8f6c6f,
                Instr::I32Load16u { .. } => 0xdc6866c6770da481,
                Instr::I32Load16uAt { .. } => 0xf194f68751968d29,
                Instr::I32Load16uOffset16 { .. } => 0xfc6373feac795559,
                Instr::I64Load8s { .. } => 0xe727f7f48695f6ad,
                Instr::I64Load8sAt { .. } => 0x9fccd4f7bd3f283f,
                Instr::I64Load8sOffset16 { .. } => 0xe865fdf1a1c55585,
                Instr::I64Load8u { .. } => 0xf78018cfa4de9cf9,
                Instr::I64Load8uAt { .. } => 0xed4846b1ee465189,
                Instr::I64Load8uOffset16 { .. } => 0xeb9c4fdbd7a69a7d,
                Instr::I64Load16s { .. } => 0xce757e747c1781e1,
                Instr::I64Load16sAt { .. } => 0x8f96d62fc6381b5b,
                Instr::I64Load16sOffset16 { .. } => 0x81747c9166be968d,
                Instr::I64Load16u { .. } => 0x9d169d9c81872e09,
                Instr::I64Load16uAt { .. } => 0x9ff242a4f7087a3b,
                Instr::I64Load16uOffset16 { .. } => 0x8a58890d2d2e95fd,
                Instr::I64Load32s { .. } => 0xc7b0ed9c7dd80abb,
                Instr::I64Load32sAt { .. } => 0xd22e5e85c5df8b81,
                Instr::I64Load32sOffset16 { .. } => 0xfe197c431899c773,
                Instr::I64Load32u { .. } => 0xec214adc8d89b335,
                Instr::I64Load32uAt { .. } => 0x8546452698268a41,
                Instr::I64Load32uOffset16 { .. } => 0x900023566f7219db,
                Instr::I32Store { .. } => 0x89b4696626e6200f,
                Instr::I32StoreOffset16 { .. } => 0xa9624220aa646c45,
                Instr::I32StoreOffset16Imm16 { .. } => 0xd375c7c6e96da7eb,
                Instr::I32StoreAt { .. } => 0x9507335cdf40a30f,
                Instr::I32StoreAtImm16 { .. } => 0xb124dcb1efb5a56f,
                Instr::I32Store8 { .. } => 0xb40f3d40e5cbc63f,
                Instr::I32Store8Offset16 { .. } => 0xb7784c5f610fa6b9,
                Instr::I32Store8Offset16Imm { .. } => 0xb1b94e6edc784d75,
                Instr::I32Store8At { .. } => 0xc114292b9396fca1,
                Instr::I32Store8AtImm { .. } => 0xf958f99a724d3fa9,
                Instr::I32Store16 { .. } => 0xf4db4b8b777ba485,
                Instr::I32Store16Offset16 { .. } => 0x917265d951560b9f,
                Instr::I32Store16Offset16Imm { .. } => 0x8e59e4b976ddd5c9,
                Instr::I32Store16At { .. } => 0x85e34459fca92a63,
                Instr::I32Store16AtImm { .. } => 0xffca87a7a28dcaaf,
                Instr::I64Store { .. } => 0xaa0cfbf1401da505,
                Instr::I64StoreOffset16 { .. } => 0xde11b832af36e2c3,
                Instr::I64StoreOffset16Imm16 { .. } => 0x93a03ca4c630054d,
                Instr::I64StoreAt { .. } => 0x8b7be36a892dbe9f,
                Instr::I64StoreAtImm16 { .. } => 0xa7164db75f5ffc79,
                Instr::I64Store8 { .. } => 0xb16fc3bd7fcf8229,
                Instr::I64Store8Offset16 { .. } => 0xf5324129bf7f4299,
                Instr::I64Store8Offset16Imm { .. } => 0xeb1df0108fb325c1,
                Instr::I64Store8At { .. } => 0xcc72df888ac47c3f,
                Instr::I64Store8AtImm { .. } => 0x90e2c84d2be4491b,
                Instr::I64Store16 { .. } => 0xa670b61daad1097f,
                Instr::I64Store16Offset16 { .. } => 0xd09b793649e2dc69,
                Instr::I64Store16Offset16Imm { .. } => 0xc5733c19fee00329,
                Instr::I64Store16At { .. } => 0xc3471d0e7d859cdd,
                Instr::I64Store16AtImm { .. } => 0xa27e4cfa22b0d101,
                Instr::I64Store32 { .. } => 0xacade9332186dab9,
                Instr::I64Store32Offset16 { .. } => 0xb5777e453e6429dd,
                Instr::I64Store32Offset16Imm16 { .. } => 0xc8435df9b5285e43,
                Instr::I64Store32At { .. } => 0xb1cb0f6ea058bbbb,
                Instr::I64Store32AtImm16 { .. } => 0x8f79394f20bbda89,
                Instr::F32Store { .. } => 0xd6df58b0ab76e99f,
                Instr::F32StoreOffset16 { .. } => 0xff1461bc14215f77,
                Instr::F32StoreAt { .. } => 0xd2f62bd6fa3c90b9,
                Instr::F64Store { .. } => 0xda484e6b7bd8d5db,
                Instr::F64StoreOffset16 { .. } => 0xac6256a3ca2605cb,
                Instr::F64StoreAt { .. } => 0xe366beba3742040b,
                ////////////////////////////////////////////////////////////////////////
                Instr::I32Eq { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0x9aa2499f95dc3711
                }
                Instr::I32EqImm16 { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0x92ce6da978fdc40f
                }
                Instr::I64Eq { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xda860a17cb3b1a8b
                }
                Instr::I64EqImm16 { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0x89c423624314bf89
                }
                Instr::I32Ne { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xcbad0daca146769f
                }
                Instr::I32NeImm16 { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xdca831c7fde0f85f
                }
                Instr::I64Ne { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xb0b2865912833697
                }
                Instr::I64NeImm16 { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xbafcb515ea4df971
                }
                Instr::I32LtS { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xbfbff0c826a235f1
                }
                Instr::I32LtU { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0x9081189b58e72897
                }
                Instr::I32LtSImm16 { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0x96d3c1dc900e1187
                }
                Instr::I32LtUImm16 { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0x9025ddff9d4de1b9
                }
                Instr::I64LtS { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xffe594b2eb58493d
                }
                Instr::I64LtU { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0x9181211dcc10809b
                }
                Instr::I64LtSImm16 { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xebbe15881dcd9e57
                }
                Instr::I64LtUImm16 { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xfd542ad322324a27
                }
                Instr::I32GtS { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xe36a0bdacb5debf3
                }
                Instr::I32GtU { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xa5deeacd3be1c44b
                }
                Instr::I32GtSImm16 { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0x9a7f06186b3795c3
                }
                Instr::I32GtUImm16 { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xaaf1e009bd26fd7b
                }
                Instr::I64GtS { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xc548cf95ce91a7b5
                }
                Instr::I64GtU { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xa68907e0fab9fb93
                }
                Instr::I64GtSImm16 { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xf62c848e8eef17e9
                }
                Instr::I64GtUImm16 { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0x8a5c73b85ba194e1
                }
                Instr::I32LeS { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xbc5f6381dd176bb1
                }
                Instr::I32LeU { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xf7f780c2e00e18af
                }
                Instr::I32LeSImm16 { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xc79fde79bd40aa77
                }
                Instr::I32LeUImm16 { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xf9cc6a0ad9269149
                }
                Instr::I64LeS { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xc7fac5fcb8f8ed55
                }
                Instr::I64LeU { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xc1560dd513285d29
                }
                Instr::I64LeSImm16 { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0x99c89d3bbb73545d
                }
                Instr::I64LeUImm16 { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xa3524cb19ca06bb5
                }
                Instr::I32GeS { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0x98bef5ced76c7645
                }
                Instr::I32GeU { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xd53ed9432d2a8143
                }
                Instr::I32GeSImm16 { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xfee355988dee53db
                }
                Instr::I32GeUImm16 { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xbb62bd7c6348e5c3
                }
                Instr::I64GeS { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xd8f40b0c313c453d
                }
                Instr::I64GeU { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0x98bfaac3f19897e1
                }
                Instr::I64GeSImm16 { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0x8956eaaa98c2e647
                }
                Instr::I64GeUImm16 { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xe11e8b930ba0afed
                }
                ////////////////////////////////////////////////////////////////////////
                Instr::F32Eq { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xc3587b028ec7b7d7
                }
                Instr::F64Eq { result, lhs, rhs } =>
                {
                    self.get_register(lhs).to_bits() ^ 0x90fa962604933679
                }
                Instr::F32Ne { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xe8b028b8a40b6323
                }
                Instr::F64Ne { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xe511c632ed75d0ad 
                }
                Instr::F32Lt { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xafe8adc3497d922f 
                }
                Instr::F64Lt { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xa24d51fe3b08563d 
                }
                Instr::F32Le { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xa9470d623de1df2f
                }
                Instr::F64Le { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xe5e889a7f2d74d67
                }
                Instr::F32Gt { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0x8cbe5aa7efd2dac5
                }
                Instr::F64Gt { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xa2b5a501d74cc69b 
                }
                Instr::F32Ge { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0x9103bfb43045fc5b
                }
                Instr::F64Ge { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xe4832a9c5a4a0741 
                }
                ////////////////////////////////////////////////////////////////////////
                Instr::I32Clz { result, input } => 
                {
                    self.get_register(input).to_bits() ^ 0xd0b363eee33e2a75
                }
                Instr::I64Clz { result, input } => 
                {
                    self.get_register(input).to_bits() ^ 0xbb13e80b90e6d539
                }
                Instr::I32Ctz { result, input } => 
                {
                    self.get_register(input).to_bits() ^ 0xa867670e58678389
                }
                Instr::I64Ctz { result, input } => 
                {
                    self.get_register(input).to_bits() ^ 0x8ad83f5db31d4957
                }
                Instr::I32Popcnt { result, input } => 
                {
                    self.get_register(input).to_bits() ^ 0xd8ad8c4a45f7cd09
                }
                Instr::I64Popcnt { result, input } => 
                {
                    self.get_register(input).to_bits() ^ 0xb9603f856bc14e5b
                }
                Instr::I32Add { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xa1f888c0cefc7b6d
                }
                Instr::I64Add { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xba1adc988a80490f
                }
                Instr::I32AddImm16 { result, lhs, rhs } => {
                    self.get_register(lhs).to_bits() ^ 0x8bc5e0c56da6ee3d
                }
                Instr::I64AddImm16 { result, lhs, rhs } => {
                    self.get_register(lhs).to_bits() ^ 0xf75f1d741e812869
                }
                Instr::I32Sub { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xf095a662a345025f
                }
                Instr::I64Sub { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xb251e2585fd105c7
                }
                Instr::I32Mul { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xc36cfc74fa61f2b3
                }
                Instr::I64Mul { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xe9fe8ad570b71a99
                }
                Instr::I32MulImm16 { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0x99c04a0680397c59
                }
                Instr::I64MulImm16 { result, lhs, rhs } => {
                    self.get_register(lhs).to_bits() ^ 0xa461c2db76abc31f
                }
                Instr::I32DivS { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xd8e9ed1b036c4299
                }
                Instr::I64DivS { result, lhs, rhs } =>
                { 
                    self.get_register(lhs).to_bits() ^ 0xc18a29741fec7821
                }
                Instr::I32DivU { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0x98f910b2a2344797
                }
                Instr::I64DivU { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xe4e5f443dafb6781
                }
                Instr::I32RemS { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0x9f03cbda2aa5fa45
                }
                Instr::I64RemS { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xccf3ffab51808eaf
                }
                Instr::I32RemU { result, lhs, rhs } =>
                { 
                    self.get_register(lhs).to_bits() ^ 0xc543d9e99bfe04db
                }
                Instr::I64RemU { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0x9f9e5bd14453abf7
                }
                ////////////////////////////////////////////////////////////////////////
                Instr::I32And { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xda40caeb3a552221
                }
                Instr::I32AndEqz { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xdae0c4aaf21a2375
                }
                Instr::I32AndEqzImm16 { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xe3b2a67a5da4fa6b
                }
                Instr::I32AndImm16 { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0x8698e382452765f1
                }
                Instr::I64And { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xf96e3bc1640f67cd
                }
                Instr::I64AndImm16 { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xcb4730c03868c6c9
                }
                Instr::I32Or { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xfe54c1a4cc88dbfd
                }
                Instr::I32OrEqz { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0x81c4ae5533789a77
                }
                Instr::I32OrEqzImm16 { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xd5a79e8c5ff0d4f7
                }
                Instr::I32OrImm16 { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0x907c4d22bec999ad
                }
                Instr::I64Or { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xdc758f1076325dcf
                }
                Instr::I64OrImm16 { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xcea9b6298da032eb
                }
                Instr::I32Xor { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0x820cea6beee5132b
                }
                Instr::I32XorEqz { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xfb5646a229eba923
                }
                Instr::I32XorEqzImm16 { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xa3d2eee0dbef491f
                }
                Instr::I32XorImm16 { result, lhs, rhs } =>
                {
                    self.get_register(lhs).to_bits() ^ 0xe0802ae028ddf527
                }
                Instr::I64Xor { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xcf8b3ddb8044776f
                }
                Instr::I64XorImm16 { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xa84e47947f7723ad
                }
                Instr::I32Shl { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0x997deb70c648fa15
                }
                Instr::I64Shl { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0x80f706f88d804a25
                }
                Instr::I32ShrU { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0x949b3946cd09a095
                }
                Instr::I64ShrU { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xbc8d8ae8fafe0cb5
                }
                Instr::I32ShrS { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xe9925858193a7679
                }
                Instr::I64ShrS { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xfb846cd977392cf3
                }
                Instr::I32Rotl { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xdfd1ecfe45e3e365
                }
                Instr::I64Rotl { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xc2c0280be48f6e2b
                }
                Instr::I32Rotr { result, lhs, rhs } =>
                { 
                    self.get_register(lhs).to_bits() ^ 0xfce450167abfef91
                }
                Instr::I64Rotr { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xd35af32013838db5
                }
                ////////////////////////////////////////////////////////////////////////
                Instr::F32Abs { result, input } => 
                {
                    self.get_register(input).to_bits() ^ 0xcd5c2fff391d82cb
                }
                Instr::F64Abs { result, input } => 
                {
                    self.get_register(input).to_bits() ^ 0xc4736057bf6ce827
                }
                Instr::F32Neg { result, input } => 
                {
                    self.get_register(input).to_bits() ^ 0xd366f959bf938435
                }
                Instr::F64Neg { result, input } => 
                {
                    self.get_register(input).to_bits() ^ 0x8c01a158032456c5
                }
                Instr::F32Ceil { result, input } => 
                {
                    self.get_register(input).to_bits() ^ 0xf5684f567a1e5c81
                }
                Instr::F64Ceil { result, input } => 
                {
                    self.get_register(input).to_bits() ^ 0xbc6729b56b5bf64f
                }
                Instr::F32Floor { result, input } => 
                {
                    self.get_register(input).to_bits() ^ 0xc3397446971c7b1b
                }
                Instr::F64Floor { result, input } => 
                {
                    self.get_register(input).to_bits() ^ 0xc21648fabc149443 
                }
                Instr::F32Trunc { result, input } => 
                {
                    self.get_register(input).to_bits() ^ 0x930c87d1457a0b2f
                }
                Instr::F64Trunc { result, input } => 
                {
                    self.get_register(input).to_bits() ^ 0xc457947a5515448d 
                }
                Instr::F32Nearest { result, input } => 
                {
                    self.get_register(input).to_bits() ^ 0xdcbd8018d58a0133 
                }
                Instr::F64Nearest { result, input } => 
                {
                    self.get_register(input).to_bits() ^ 0xe719d229d8dc9d11
                }
                Instr::F32Sqrt { result, input } =>
                {
                    self.get_register(input).to_bits() ^ 0x8e031a9674797f6f
                }
                Instr::F64Sqrt { result, input } => 
                {
                    self.get_register(input).to_bits() ^ 0xede241e1bdbf8add 
                }
                ////////////////////////////////////////////////////////////////////////
                Instr::F32Add { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xc0246fd5a4fa2569 
                }
                Instr::F64Add { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xae61b186b8d627b1 
                }
                Instr::F32Sub { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xf37398a1108c36cb
                }
                Instr::F64Sub { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xaaf86176c0dc89f5
                }
                Instr::F32Mul { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xbe5eb79b83c0c7b1 
                }
                Instr::F64Mul { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0x9b200d1c1640bf0d 
                }
                Instr::F32Div { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xd22d29503c878647 
                }
                Instr::F64Div { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0x91b08c54e524bb09 
                }
                Instr::F32Min { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xf83af276dd4b617f 
                }
                Instr::F64Min { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xeb5d7d82375f7be7
                }
                Instr::F32Max { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0x8f4d06f60f1c84fb 
                }
                Instr::F64Max { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xb24f6877be71ebd5 
                }
                Instr::F32Copysign { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xec122620e993dfcd
                }
                Instr::F64Copysign { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0xf27f1850006566c9 
                }
                Instr::F32CopysignImm { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0x94d19450082a4ce9 
                }
                Instr::F64CopysignImm { result, lhs, rhs } => 
                {
                    self.get_register(lhs).to_bits() ^ 0x84b094c8c0503805
                }
                ////////////////////////////////////////////////////////////////////////
                Instr::I32WrapI64 { result, input } => 
                {
                    self.get_register(input).to_bits() ^ 0xd7348da1051ffdf5
                }
                Instr::I32TruncF32S { result, input } =>
                {
                    self.get_register(input).to_bits() ^ 0xa8edf1813c31175b
                }
                Instr::I32TruncF32U { result, input } => 
                {
                    self.get_register(input).to_bits() ^ 0xf980305a8ba3be0f
                }
                Instr::I32TruncF64S { result, input } => 
                {
                    self.get_register(input).to_bits() ^ 0xb982c8f45dcd5731
                }
                Instr::I32TruncF64U { result, input } =>
                {
                    self.get_register(input).to_bits() ^ 0x9ca1670d1e934f45 
                }
                Instr::I64TruncF32S { result, input } => 
                {
                    self.get_register(input).to_bits() ^ 0xeb2506c7a7cfe6f7 
                }
                Instr::I64TruncF32U { result, input } => 
                {
                    self.get_register(input).to_bits() ^ 0xa230e0381f36668d
                }
                Instr::I64TruncF64S { result, input } => 
                {
                    self.get_register(input).to_bits() ^ 0xce02765ab94df325
                }
                Instr::I64TruncF64U { result, input } => 
                {
                    self.get_register(input).to_bits() ^ 0xb39253799e21a72d
                }
                Instr::I32TruncSatF32S { result, input } => 
                {
                    self.get_register(input).to_bits() ^ 0xa164fb50eec581d3
                }
                Instr::I32TruncSatF32U { result, input } => 
                {
                    self.get_register(input).to_bits() ^ 0xabac89637bdb1d8f
                }
                Instr::I32TruncSatF64S { result, input } => 
                {
                    self.get_register(input).to_bits() ^ 0xe8b8c4421046aedd 
                }
                Instr::I32TruncSatF64U { result, input } => 
                {
                    self.get_register(input).to_bits() ^ 0x91c87015a56a944d
                }
                Instr::I64TruncSatF32S { result, input } => 
                {
                    self.get_register(input).to_bits() ^ 0xb909e169382afddd
                }
                Instr::I64TruncSatF32U { result, input } => 
                {
                    self.get_register(input).to_bits() ^ 0xc6f884d2705bf2d3
                }
                Instr::I64TruncSatF64S { result, input } =>
                {
                    self.get_register(input).to_bits() ^ 0xa5e8386963664fa3
                }
                Instr::I64TruncSatF64U { result, input } =>
                {
                    self.get_register(input).to_bits() ^ 0xa43800f9e4975aff
                } 
                Instr::I32Extend8S { result, input } => 
                {
                    self.get_register(input).to_bits() ^ 0xdecfc0dc5cb809af
                }
                Instr::I32Extend16S { result, input } => 
                {
                    self.get_register(input).to_bits() ^ 0xcdf6bb7756026125
                }
                Instr::I64Extend8S { result, input } => 
                {
                    self.get_register(input).to_bits() ^ 0xb906176cee2380bf
                }
                Instr::I64Extend16S { result, input } => 
                {
                    self.get_register(input).to_bits() ^ 0xf0382669ed7a55f1
                }
                Instr::I64Extend32S { result, input } => 
                {
                    self.get_register(input).to_bits() ^ 0xb61b2de5652d06e9
                }
                Instr::F32DemoteF64 { result, input } => 
                {
                    self.get_register(input).to_bits() ^ 0xbf82e5dd4495233b
                }
                Instr::F64PromoteF32 { result, input } => 
                {
                    self.get_register(input).to_bits() ^ 0xf42a79d7ed7c17c
                }
                Instr::F32ConvertI32S { result, input } => 
                {
                    self.get_register(input).to_bits() ^ 0x9e65030287165e29
                }
                Instr::F32ConvertI32U { result, input } => 
                {
                    self.get_register(input).to_bits() ^ 0xe244f0acd2209f0b
                }
                Instr::F32ConvertI64S { result, input } => 
                {
                    self.get_register(input).to_bits() ^ 0xd007d6d9333c7405
                }
                Instr::F32ConvertI64U { result, input } => 
                {
                    self.get_register(input).to_bits() ^ 0xf31a7af87a7b7f61
                }
                Instr::F64ConvertI32S { result, input } => 
                {
                    self.get_register(input).to_bits() ^ 0xbef1a0dfa7540b4d
                }
                Instr::F64ConvertI32U { result, input } => 
                {
                    self.get_register(input).to_bits() ^ 0xa168200e59e18dcd
                }
                Instr::F64ConvertI64S { result, input } => 
                {
                    self.get_register(input).to_bits() ^ 0xb3a2d5946ee565e3
                }
                Instr::F64ConvertI64U { result, input } => 
                {
                    self.get_register(input).to_bits() ^ 0x92ad2f2873e8fbc5
                }
                _ => 0xf360371a61b48ca1,
            };

            store.inner.update_runtime_signature(instr_prime);

            match instr {
                Instr::Trap { trap_code } => self.execute_trap(trap_code)?,
                Instr::ConsumeFuel { block_fuel } => {
                    self.execute_consume_fuel(&mut store.inner, block_fuel)?
                }
                Instr::Return => {
                    forward_return!(self.execute_return(&mut store.inner))
                }
                Instr::ReturnReg { value } => {
                    forward_return!(self.execute_return_reg(&mut store.inner, value))
                }
                Instr::ReturnReg2 { values } => {
                    forward_return!(self.execute_return_reg2(&mut store.inner, values))
                }
                Instr::ReturnReg3 { values } => {
                    forward_return!(self.execute_return_reg3(&mut store.inner, values))
                }
                Instr::ReturnImm32 { value } => {
                    forward_return!(self.execute_return_imm32(&mut store.inner, value))
                }
                Instr::ReturnI64Imm32 { value } => {
                    forward_return!(self.execute_return_i64imm32(&mut store.inner, value))
                }
                Instr::ReturnF64Imm32 { value } => {
                    forward_return!(self.execute_return_f64imm32(&mut store.inner, value))
                }
                Instr::ReturnSpan { values } => {
                    forward_return!(self.execute_return_span(&mut store.inner, values))
                }
                Instr::ReturnMany { values } => {
                    forward_return!(self.execute_return_many(&mut store.inner, values))
                }
                Instr::ReturnNez { condition } => {
                    forward_return!(self.execute_return_nez(&mut store.inner, condition))
                }
                Instr::ReturnNezReg { condition, value } => {
                    forward_return!(self.execute_return_nez_reg(&mut store.inner, condition, value))
                }
                Instr::ReturnNezReg2 { condition, values } => {
                    forward_return!(self.execute_return_nez_reg2(
                        &mut store.inner,
                        condition,
                        values
                    ))
                }
                Instr::ReturnNezImm32 { condition, value } => {
                    forward_return!(self.execute_return_nez_imm32(
                        &mut store.inner,
                        condition,
                        value
                    ))
                }
                Instr::ReturnNezI64Imm32 { condition, value } => {
                    forward_return!(self.execute_return_nez_i64imm32(
                        &mut store.inner,
                        condition,
                        value
                    ))
                }
                Instr::ReturnNezF64Imm32 { condition, value } => {
                    forward_return!(self.execute_return_nez_f64imm32(
                        &mut store.inner,
                        condition,
                        value
                    ))
                }
                Instr::ReturnNezSpan { condition, values } => {
                    forward_return!(self.execute_return_nez_span(
                        &mut store.inner,
                        condition,
                        values
                    ))
                }
                Instr::ReturnNezMany { condition, values } => {
                    forward_return!(self.execute_return_nez_many(
                        &mut store.inner,
                        condition,
                        values
                    ))
                }
                Instr::Branch { offset } => self.execute_branch(offset),
                Instr::BranchTable0 { index, len_targets } => {
                    self.execute_branch_table_0(index, len_targets)
                }
                Instr::BranchTable1 { index, len_targets } => {
                    self.execute_branch_table_1(index, len_targets)
                }
                Instr::BranchTable2 { index, len_targets } => {
                    self.execute_branch_table_2(index, len_targets)
                }
                Instr::BranchTable3 { index, len_targets } => {
                    self.execute_branch_table_3(index, len_targets)
                }
                Instr::BranchTableSpan { index, len_targets } => {
                    self.execute_branch_table_span(index, len_targets)
                }
                Instr::BranchTableMany { index, len_targets } => {
                    self.execute_branch_table_many(index, len_targets)
                }
                Instr::BranchCmpFallback { lhs, rhs, params } => {
                    self.execute_branch_cmp_fallback(lhs, rhs, params)
                }
                Instr::BranchI32And { lhs, rhs, offset } => {
                    self.execute_branch_i32_and(lhs, rhs, offset)
                }
                Instr::BranchI32AndImm { lhs, rhs, offset } => {
                    self.execute_branch_i32_and_imm(lhs, rhs, offset)
                }
                Instr::BranchI32Or { lhs, rhs, offset } => {
                    self.execute_branch_i32_or(lhs, rhs, offset)
                }
                Instr::BranchI32OrImm { lhs, rhs, offset } => {
                    self.execute_branch_i32_or_imm(lhs, rhs, offset)
                }
                Instr::BranchI32Xor { lhs, rhs, offset } => {
                    self.execute_branch_i32_xor(lhs, rhs, offset)
                }
                Instr::BranchI32XorImm { lhs, rhs, offset } => {
                    self.execute_branch_i32_xor_imm(lhs, rhs, offset)
                }
                Instr::BranchI32AndEqz { lhs, rhs, offset } => {
                    self.execute_branch_i32_and_eqz(lhs, rhs, offset)
                }
                Instr::BranchI32AndEqzImm { lhs, rhs, offset } => {
                    self.execute_branch_i32_and_eqz_imm(lhs, rhs, offset)
                }
                Instr::BranchI32OrEqz { lhs, rhs, offset } => {
                    self.execute_branch_i32_or_eqz(lhs, rhs, offset)
                }
                Instr::BranchI32OrEqzImm { lhs, rhs, offset } => {
                    self.execute_branch_i32_or_eqz_imm(lhs, rhs, offset)
                }
                Instr::BranchI32XorEqz { lhs, rhs, offset } => {
                    self.execute_branch_i32_xor_eqz(lhs, rhs, offset)
                }
                Instr::BranchI32XorEqzImm { lhs, rhs, offset } => {
                    self.execute_branch_i32_xor_eqz_imm(lhs, rhs, offset)
                }
                Instr::BranchI32Eq { lhs, rhs, offset } => {
                    self.execute_branch_i32_eq(lhs, rhs, offset)
                }
                Instr::BranchI32EqImm { lhs, rhs, offset } => {
                    self.execute_branch_i32_eq_imm(lhs, rhs, offset)
                }
                Instr::BranchI32Ne { lhs, rhs, offset } => {
                    self.execute_branch_i32_ne(lhs, rhs, offset)
                }
                Instr::BranchI32NeImm { lhs, rhs, offset } => {
                    self.execute_branch_i32_ne_imm(lhs, rhs, offset)
                }
                Instr::BranchI32LtS { lhs, rhs, offset } => {
                    self.execute_branch_i32_lt_s(lhs, rhs, offset)
                }
                Instr::BranchI32LtSImm { lhs, rhs, offset } => {
                    self.execute_branch_i32_lt_s_imm(lhs, rhs, offset)
                }
                Instr::BranchI32LtU { lhs, rhs, offset } => {
                    self.execute_branch_i32_lt_u(lhs, rhs, offset)
                }
                Instr::BranchI32LtUImm { lhs, rhs, offset } => {
                    self.execute_branch_i32_lt_u_imm(lhs, rhs, offset)
                }
                Instr::BranchI32LeS { lhs, rhs, offset } => {
                    self.execute_branch_i32_le_s(lhs, rhs, offset)
                }
                Instr::BranchI32LeSImm { lhs, rhs, offset } => {
                    self.execute_branch_i32_le_s_imm(lhs, rhs, offset)
                }
                Instr::BranchI32LeU { lhs, rhs, offset } => {
                    self.execute_branch_i32_le_u(lhs, rhs, offset)
                }
                Instr::BranchI32LeUImm { lhs, rhs, offset } => {
                    self.execute_branch_i32_le_u_imm(lhs, rhs, offset)
                }
                Instr::BranchI32GtS { lhs, rhs, offset } => {
                    self.execute_branch_i32_gt_s(lhs, rhs, offset)
                }
                Instr::BranchI32GtSImm { lhs, rhs, offset } => {
                    self.execute_branch_i32_gt_s_imm(lhs, rhs, offset)
                }
                Instr::BranchI32GtU { lhs, rhs, offset } => {
                    self.execute_branch_i32_gt_u(lhs, rhs, offset)
                }
                Instr::BranchI32GtUImm { lhs, rhs, offset } => {
                    self.execute_branch_i32_gt_u_imm(lhs, rhs, offset)
                }
                Instr::BranchI32GeS { lhs, rhs, offset } => {
                    self.execute_branch_i32_ge_s(lhs, rhs, offset)
                }
                Instr::BranchI32GeSImm { lhs, rhs, offset } => {
                    self.execute_branch_i32_ge_s_imm(lhs, rhs, offset)
                }
                Instr::BranchI32GeU { lhs, rhs, offset } => {
                    self.execute_branch_i32_ge_u(lhs, rhs, offset)
                }
                Instr::BranchI32GeUImm { lhs, rhs, offset } => {
                    self.execute_branch_i32_ge_u_imm(lhs, rhs, offset)
                }
                Instr::BranchI64Eq { lhs, rhs, offset } => {
                    self.execute_branch_i64_eq(lhs, rhs, offset)
                }
                Instr::BranchI64EqImm { lhs, rhs, offset } => {
                    self.execute_branch_i64_eq_imm(lhs, rhs, offset)
                }
                Instr::BranchI64Ne { lhs, rhs, offset } => {
                    self.execute_branch_i64_ne(lhs, rhs, offset)
                }
                Instr::BranchI64NeImm { lhs, rhs, offset } => {
                    self.execute_branch_i64_ne_imm(lhs, rhs, offset)
                }
                Instr::BranchI64LtS { lhs, rhs, offset } => {
                    self.execute_branch_i64_lt_s(lhs, rhs, offset)
                }
                Instr::BranchI64LtSImm { lhs, rhs, offset } => {
                    self.execute_branch_i64_lt_s_imm(lhs, rhs, offset)
                }
                Instr::BranchI64LtU { lhs, rhs, offset } => {
                    self.execute_branch_i64_lt_u(lhs, rhs, offset)
                }
                Instr::BranchI64LtUImm { lhs, rhs, offset } => {
                    self.execute_branch_i64_lt_u_imm(lhs, rhs, offset)
                }
                Instr::BranchI64LeS { lhs, rhs, offset } => {
                    self.execute_branch_i64_le_s(lhs, rhs, offset)
                }
                Instr::BranchI64LeSImm { lhs, rhs, offset } => {
                    self.execute_branch_i64_le_s_imm(lhs, rhs, offset)
                }
                Instr::BranchI64LeU { lhs, rhs, offset } => {
                    self.execute_branch_i64_le_u(lhs, rhs, offset)
                }
                Instr::BranchI64LeUImm { lhs, rhs, offset } => {
                    self.execute_branch_i64_le_u_imm(lhs, rhs, offset)
                }
                Instr::BranchI64GtS { lhs, rhs, offset } => {
                    self.execute_branch_i64_gt_s(lhs, rhs, offset)
                }
                Instr::BranchI64GtSImm { lhs, rhs, offset } => {
                    self.execute_branch_i64_gt_s_imm(lhs, rhs, offset)
                }
                Instr::BranchI64GtU { lhs, rhs, offset } => {
                    self.execute_branch_i64_gt_u(lhs, rhs, offset)
                }
                Instr::BranchI64GtUImm { lhs, rhs, offset } => {
                    self.execute_branch_i64_gt_u_imm(lhs, rhs, offset)
                }
                Instr::BranchI64GeS { lhs, rhs, offset } => {
                    self.execute_branch_i64_ge_s(lhs, rhs, offset)
                }
                Instr::BranchI64GeSImm { lhs, rhs, offset } => {
                    self.execute_branch_i64_ge_s_imm(lhs, rhs, offset)
                }
                Instr::BranchI64GeU { lhs, rhs, offset } => {
                    self.execute_branch_i64_ge_u(lhs, rhs, offset)
                }
                Instr::BranchI64GeUImm { lhs, rhs, offset } => {
                    self.execute_branch_i64_ge_u_imm(lhs, rhs, offset)
                }
                Instr::BranchF32Eq { lhs, rhs, offset } => {
                    self.execute_branch_f32_eq(lhs, rhs, offset)
                }
                Instr::BranchF32Ne { lhs, rhs, offset } => {
                    self.execute_branch_f32_ne(lhs, rhs, offset)
                }
                Instr::BranchF32Lt { lhs, rhs, offset } => {
                    self.execute_branch_f32_lt(lhs, rhs, offset)
                }
                Instr::BranchF32Le { lhs, rhs, offset } => {
                    self.execute_branch_f32_le(lhs, rhs, offset)
                }
                Instr::BranchF32Gt { lhs, rhs, offset } => {
                    self.execute_branch_f32_gt(lhs, rhs, offset)
                }
                Instr::BranchF32Ge { lhs, rhs, offset } => {
                    self.execute_branch_f32_ge(lhs, rhs, offset)
                }
                Instr::BranchF64Eq { lhs, rhs, offset } => {
                    self.execute_branch_f64_eq(lhs, rhs, offset)
                }
                Instr::BranchF64Ne { lhs, rhs, offset } => {
                    self.execute_branch_f64_ne(lhs, rhs, offset)
                }
                Instr::BranchF64Lt { lhs, rhs, offset } => {
                    self.execute_branch_f64_lt(lhs, rhs, offset)
                }
                Instr::BranchF64Le { lhs, rhs, offset } => {
                    self.execute_branch_f64_le(lhs, rhs, offset)
                }
                Instr::BranchF64Gt { lhs, rhs, offset } => {
                    self.execute_branch_f64_gt(lhs, rhs, offset)
                }
                Instr::BranchF64Ge { lhs, rhs, offset } => {
                    self.execute_branch_f64_ge(lhs, rhs, offset)
                }
                Instr::Copy { result, value } => self.execute_copy(result, value),
                Instr::Copy2 { results, values } => self.execute_copy_2(results, values),
                Instr::CopyImm32 { result, value } => self.execute_copy_imm32(result, value),
                Instr::CopyI64Imm32 { result, value } => self.execute_copy_i64imm32(result, value),
                Instr::CopyF64Imm32 { result, value } => self.execute_copy_f64imm32(result, value),
                Instr::CopySpan {
                    results,
                    values,
                    len,
                } => self.execute_copy_span(results, values, len),
                Instr::CopySpanNonOverlapping {
                    results,
                    values,
                    len,
                } => self.execute_copy_span_non_overlapping(results, values, len),
                Instr::CopyMany { results, values } => self.execute_copy_many(results, values),
                Instr::CopyManyNonOverlapping { results, values } => {
                    self.execute_copy_many_non_overlapping(results, values)
                }
                Instr::ReturnCallInternal0 { func } => {
                    self.execute_return_call_internal_0(&mut store.inner, EngineFunc::from(func))?
                }
                Instr::ReturnCallInternal { func } => {
                    self.execute_return_call_internal(&mut store.inner, EngineFunc::from(func))?
                }
                Instr::ReturnCallImported0 { func } => {
                    self.execute_return_call_imported_0::<T>(store, func)?
                }
                Instr::ReturnCallImported { func } => {
                    self.execute_return_call_imported::<T>(store, func)?
                }
                Instr::ReturnCallIndirect0 { func_type } => {
                    self.execute_return_call_indirect_0::<T>(store, func_type)?
                }
                Instr::ReturnCallIndirect0Imm16 { func_type } => {
                    self.execute_return_call_indirect_0_imm16::<T>(store, func_type)?
                }
                Instr::ReturnCallIndirect { func_type } => {
                    self.execute_return_call_indirect::<T>(store, func_type)?
                }
                Instr::ReturnCallIndirectImm16 { func_type } => {
                    self.execute_return_call_indirect_imm16::<T>(store, func_type)?
                }
                Instr::CallInternal0 { results, func } => {
                    self.execute_call_internal_0(&mut store.inner, results, EngineFunc::from(func))?
                }
                Instr::CallInternal { results, func } => {
                    self.execute_call_internal(&mut store.inner, results, EngineFunc::from(func))?
                }
                Instr::CallImported0 { results, func } => {
                    self.execute_call_imported_0::<T>(store, results, func)?
                }
                Instr::CallImported { results, func } => {
                    self.execute_call_imported::<T>(store, results, func)?
                }
                Instr::CallIndirect0 { results, func_type } => {
                    self.execute_call_indirect_0::<T>(store, results, func_type)?
                }
                Instr::CallIndirect0Imm16 { results, func_type } => {
                    self.execute_call_indirect_0_imm16::<T>(store, results, func_type)?
                }
                Instr::CallIndirect { results, func_type } => {
                    self.execute_call_indirect::<T>(store, results, func_type)?
                }
                Instr::CallIndirectImm16 { results, func_type } => {
                    self.execute_call_indirect_imm16::<T>(store, results, func_type)?
                }
                Instr::Select { result, lhs } => self.execute_select(result, lhs),
                Instr::SelectImm32Rhs { result, lhs } => self.execute_select_imm32_rhs(result, lhs),
                Instr::SelectImm32Lhs { result, lhs } => self.execute_select_imm32_lhs(result, lhs),
                Instr::SelectImm32 { result, lhs } => self.execute_select_imm32(result, lhs),
                Instr::SelectI64Imm32Rhs { result, lhs } => {
                    self.execute_select_i64imm32_rhs(result, lhs)
                }
                Instr::SelectI64Imm32Lhs { result, lhs } => {
                    self.execute_select_i64imm32_lhs(result, lhs)
                }
                Instr::SelectI64Imm32 { result, lhs } => self.execute_select_i64imm32(result, lhs),
                Instr::SelectF64Imm32Rhs { result, lhs } => {
                    self.execute_select_f64imm32_rhs(result, lhs)
                }
                Instr::SelectF64Imm32Lhs { result, lhs } => {
                    self.execute_select_f64imm32_lhs(result, lhs)
                }
                Instr::SelectF64Imm32 { result, lhs } => self.execute_select_f64imm32(result, lhs),
                Instr::RefFunc { result, func } => self.execute_ref_func(result, func),
                Instr::GlobalGet { result, global } => {
                    self.execute_global_get(&store.inner, result, global)
                }
                Instr::GlobalSet { global, input } => {
                    self.execute_global_set(&mut store.inner, global, input)
                }
                Instr::GlobalSetI32Imm16 { global, input } => {
                    self.execute_global_set_i32imm16(&mut store.inner, global, input)
                }
                Instr::GlobalSetI64Imm16 { global, input } => {
                    self.execute_global_set_i64imm16(&mut store.inner, global, input)
                }
                Instr::I32Load { result, memory } => {
                    self.execute_i32_load(&store.inner, result, memory)?
                }
                Instr::I32LoadAt { result, address } => {
                    self.execute_i32_load_at(&store.inner, result, address)?
                }
                Instr::I32LoadOffset16 {
                    result,
                    ptr,
                    offset,
                } => self.execute_i32_load_offset16(result, ptr, offset)?,
                Instr::I64Load { result, memory } => {
                    self.execute_i64_load(&store.inner, result, memory)?
                }
                Instr::I64LoadAt { result, address } => {
                    self.execute_i64_load_at(&store.inner, result, address)?
                }
                Instr::I64LoadOffset16 {
                    result,
                    ptr,
                    offset,
                } => self.execute_i64_load_offset16(result, ptr, offset)?,
                Instr::F32Load { result, memory } => {
                    self.execute_f32_load(&store.inner, result, memory)?
                }
                Instr::F32LoadAt { result, address } => {
                    self.execute_f32_load_at(&store.inner, result, address)?
                }
                Instr::F32LoadOffset16 {
                    result,
                    ptr,
                    offset,
                } => self.execute_f32_load_offset16(result, ptr, offset)?,
                Instr::F64Load { result, memory } => {
                    self.execute_f64_load(&store.inner, result, memory)?
                }
                Instr::F64LoadAt { result, address } => {
                    self.execute_f64_load_at(&store.inner, result, address)?
                }
                Instr::F64LoadOffset16 {
                    result,
                    ptr,
                    offset,
                } => self.execute_f64_load_offset16(result, ptr, offset)?,
                Instr::I32Load8s { result, memory } => {
                    self.execute_i32_load8_s(&store.inner, result, memory)?
                }
                Instr::I32Load8sAt { result, address } => {
                    self.execute_i32_load8_s_at(&store.inner, result, address)?
                }
                Instr::I32Load8sOffset16 {
                    result,
                    ptr,
                    offset,
                } => self.execute_i32_load8_s_offset16(result, ptr, offset)?,
                Instr::I32Load8u { result, memory } => {
                    self.execute_i32_load8_u(&store.inner, result, memory)?
                }
                Instr::I32Load8uAt { result, address } => {
                    self.execute_i32_load8_u_at(&store.inner, result, address)?
                }
                Instr::I32Load8uOffset16 {
                    result,
                    ptr,
                    offset,
                } => self.execute_i32_load8_u_offset16(result, ptr, offset)?,
                Instr::I32Load16s { result, memory } => {
                    self.execute_i32_load16_s(&store.inner, result, memory)?
                }
                Instr::I32Load16sAt { result, address } => {
                    self.execute_i32_load16_s_at(&store.inner, result, address)?
                }
                Instr::I32Load16sOffset16 {
                    result,
                    ptr,
                    offset,
                } => self.execute_i32_load16_s_offset16(result, ptr, offset)?,
                Instr::I32Load16u { result, memory } => {
                    self.execute_i32_load16_u(&store.inner, result, memory)?
                }
                Instr::I32Load16uAt { result, address } => {
                    self.execute_i32_load16_u_at(&store.inner, result, address)?
                }
                Instr::I32Load16uOffset16 {
                    result,
                    ptr,
                    offset,
                } => self.execute_i32_load16_u_offset16(result, ptr, offset)?,
                Instr::I64Load8s { result, memory } => {
                    self.execute_i64_load8_s(&store.inner, result, memory)?
                }
                Instr::I64Load8sAt { result, address } => {
                    self.execute_i64_load8_s_at(&store.inner, result, address)?
                }
                Instr::I64Load8sOffset16 {
                    result,
                    ptr,
                    offset,
                } => self.execute_i64_load8_s_offset16(result, ptr, offset)?,
                Instr::I64Load8u { result, memory } => {
                    self.execute_i64_load8_u(&store.inner, result, memory)?
                }
                Instr::I64Load8uAt { result, address } => {
                    self.execute_i64_load8_u_at(&store.inner, result, address)?
                }
                Instr::I64Load8uOffset16 {
                    result,
                    ptr,
                    offset,
                } => self.execute_i64_load8_u_offset16(result, ptr, offset)?,
                Instr::I64Load16s { result, memory } => {
                    self.execute_i64_load16_s(&store.inner, result, memory)?
                }
                Instr::I64Load16sAt { result, address } => {
                    self.execute_i64_load16_s_at(&store.inner, result, address)?
                }
                Instr::I64Load16sOffset16 {
                    result,
                    ptr,
                    offset,
                } => self.execute_i64_load16_s_offset16(result, ptr, offset)?,
                Instr::I64Load16u { result, memory } => {
                    self.execute_i64_load16_u(&store.inner, result, memory)?
                }
                Instr::I64Load16uAt { result, address } => {
                    self.execute_i64_load16_u_at(&store.inner, result, address)?
                }
                Instr::I64Load16uOffset16 {
                    result,
                    ptr,
                    offset,
                } => self.execute_i64_load16_u_offset16(result, ptr, offset)?,
                Instr::I64Load32s { result, memory } => {
                    self.execute_i64_load32_s(&store.inner, result, memory)?
                }
                Instr::I64Load32sAt { result, address } => {
                    self.execute_i64_load32_s_at(&store.inner, result, address)?
                }
                Instr::I64Load32sOffset16 {
                    result,
                    ptr,
                    offset,
                } => self.execute_i64_load32_s_offset16(result, ptr, offset)?,
                Instr::I64Load32u { result, memory } => {
                    self.execute_i64_load32_u(&store.inner, result, memory)?
                }
                Instr::I64Load32uAt { result, address } => {
                    self.execute_i64_load32_u_at(&store.inner, result, address)?
                }
                Instr::I64Load32uOffset16 {
                    result,
                    ptr,
                    offset,
                } => self.execute_i64_load32_u_offset16(result, ptr, offset)?,
                Instr::I32Store { ptr, memory } => {
                    self.execute_i32_store(&mut store.inner, ptr, memory)?
                }
                Instr::I32StoreImm16 { ptr, memory } => {
                    self.execute_i32_store_imm16(&mut store.inner, ptr, memory)?
                }
                Instr::I32StoreOffset16 { ptr, offset, value } => {
                    self.execute_i32_store_offset16(ptr, offset, value)?
                }
                Instr::I32StoreOffset16Imm16 { ptr, offset, value } => {
                    self.execute_i32_store_offset16_imm16(ptr, offset, value)?
                }
                Instr::I32StoreAt { address, value } => {
                    self.execute_i32_store_at(&mut store.inner, address, value)?
                }
                Instr::I32StoreAtImm16 { address, value } => {
                    self.execute_i32_store_at_imm16(&mut store.inner, address, value)?
                }
                Instr::I32Store8 { ptr, memory } => {
                    self.execute_i32_store8(&mut store.inner, ptr, memory)?
                }
                Instr::I32Store8Imm { ptr, memory } => {
                    self.execute_i32_store8_imm(&mut store.inner, ptr, memory)?
                }
                Instr::I32Store8Offset16 { ptr, offset, value } => {
                    self.execute_i32_store8_offset16(ptr, offset, value)?
                }
                Instr::I32Store8Offset16Imm { ptr, offset, value } => {
                    self.execute_i32_store8_offset16_imm(ptr, offset, value)?
                }
                Instr::I32Store8At { address, value } => {
                    self.execute_i32_store8_at(&mut store.inner, address, value)?
                }
                Instr::I32Store8AtImm { address, value } => {
                    self.execute_i32_store8_at_imm(&mut store.inner, address, value)?
                }
                Instr::I32Store16 { ptr, memory } => {
                    self.execute_i32_store16(&mut store.inner, ptr, memory)?
                }
                Instr::I32Store16Imm { ptr, memory } => {
                    self.execute_i32_store16_imm(&mut store.inner, ptr, memory)?
                }
                Instr::I32Store16Offset16 { ptr, offset, value } => {
                    self.execute_i32_store16_offset16(ptr, offset, value)?
                }
                Instr::I32Store16Offset16Imm { ptr, offset, value } => {
                    self.execute_i32_store16_offset16_imm(ptr, offset, value)?
                }
                Instr::I32Store16At { address, value } => {
                    self.execute_i32_store16_at(&mut store.inner, address, value)?
                }
                Instr::I32Store16AtImm { address, value } => {
                    self.execute_i32_store16_at_imm(&mut store.inner, address, value)?
                }
                Instr::I64Store { ptr, memory } => {
                    self.execute_i64_store(&mut store.inner, ptr, memory)?
                }
                Instr::I64StoreImm16 { ptr, memory } => {
                    self.execute_i64_store_imm16(&mut store.inner, ptr, memory)?
                }
                Instr::I64StoreOffset16 { ptr, offset, value } => {
                    self.execute_i64_store_offset16(ptr, offset, value)?
                }
                Instr::I64StoreOffset16Imm16 { ptr, offset, value } => {
                    self.execute_i64_store_offset16_imm16(ptr, offset, value)?
                }
                Instr::I64StoreAt { address, value } => {
                    self.execute_i64_store_at(&mut store.inner, address, value)?
                }
                Instr::I64StoreAtImm16 { address, value } => {
                    self.execute_i64_store_at_imm16(&mut store.inner, address, value)?
                }
                Instr::I64Store8 { ptr, memory } => {
                    self.execute_i64_store8(&mut store.inner, ptr, memory)?
                }
                Instr::I64Store8Imm { ptr, memory } => {
                    self.execute_i64_store8_imm(&mut store.inner, ptr, memory)?
                }
                Instr::I64Store8Offset16 { ptr, offset, value } => {
                    self.execute_i64_store8_offset16(ptr, offset, value)?
                }
                Instr::I64Store8Offset16Imm { ptr, offset, value } => {
                    self.execute_i64_store8_offset16_imm(ptr, offset, value)?
                }
                Instr::I64Store8At { address, value } => {
                    self.execute_i64_store8_at(&mut store.inner, address, value)?
                }
                Instr::I64Store8AtImm { address, value } => {
                    self.execute_i64_store8_at_imm(&mut store.inner, address, value)?
                }
                Instr::I64Store16 { ptr, memory } => {
                    self.execute_i64_store16(&mut store.inner, ptr, memory)?
                }
                Instr::I64Store16Imm { ptr, memory } => {
                    self.execute_i64_store16_imm(&mut store.inner, ptr, memory)?
                }
                Instr::I64Store16Offset16 { ptr, offset, value } => {
                    self.execute_i64_store16_offset16(ptr, offset, value)?
                }
                Instr::I64Store16Offset16Imm { ptr, offset, value } => {
                    self.execute_i64_store16_offset16_imm(ptr, offset, value)?
                }
                Instr::I64Store16At { address, value } => {
                    self.execute_i64_store16_at(&mut store.inner, address, value)?
                }
                Instr::I64Store16AtImm { address, value } => {
                    self.execute_i64_store16_at_imm(&mut store.inner, address, value)?
                }
                Instr::I64Store32 { ptr, memory } => {
                    self.execute_i64_store32(&mut store.inner, ptr, memory)?
                }
                Instr::I64Store32Imm16 { ptr, memory } => {
                    self.execute_i64_store32_imm16(&mut store.inner, ptr, memory)?
                }
                Instr::I64Store32Offset16 { ptr, offset, value } => {
                    self.execute_i64_store32_offset16(ptr, offset, value)?
                }
                Instr::I64Store32Offset16Imm16 { ptr, offset, value } => {
                    self.execute_i64_store32_offset16_imm16(ptr, offset, value)?
                }
                Instr::I64Store32At { address, value } => {
                    self.execute_i64_store32_at(&mut store.inner, address, value)?
                }
                Instr::I64Store32AtImm16 { address, value } => {
                    self.execute_i64_store32_at_imm16(&mut store.inner, address, value)?
                }
                Instr::F32Store { ptr, memory } => {
                    self.execute_f32_store(&mut store.inner, ptr, memory)?
                }
                Instr::F32StoreOffset16 { ptr, offset, value } => {
                    self.execute_f32_store_offset16(ptr, offset, value)?
                }
                Instr::F32StoreAt { address, value } => {
                    self.execute_f32_store_at(&mut store.inner, address, value)?
                }
                Instr::F64Store { ptr, memory } => {
                    self.execute_f64_store(&mut store.inner, ptr, memory)?
                }
                Instr::F64StoreOffset16 { ptr, offset, value } => {
                    self.execute_f64_store_offset16(ptr, offset, value)?
                }
                Instr::F64StoreAt { address, value } => {
                    self.execute_f64_store_at(&mut store.inner, address, value)?
                }
                Instr::I32Eq { result, lhs, rhs } => self.execute_i32_eq(result, lhs, rhs),
                Instr::I32EqImm16 { result, lhs, rhs } => {
                    self.execute_i32_eq_imm16(result, lhs, rhs)
                }
                Instr::I32Ne { result, lhs, rhs } => self.execute_i32_ne(result, lhs, rhs),
                Instr::I32NeImm16 { result, lhs, rhs } => {
                    self.execute_i32_ne_imm16(result, lhs, rhs)
                }
                Instr::I32LtS { result, lhs, rhs } => self.execute_i32_lt_s(result, lhs, rhs),
                Instr::I32LtSImm16 { result, lhs, rhs } => {
                    self.execute_i32_lt_s_imm16(result, lhs, rhs)
                }
                Instr::I32LtU { result, lhs, rhs } => self.execute_i32_lt_u(result, lhs, rhs),
                Instr::I32LtUImm16 { result, lhs, rhs } => {
                    self.execute_i32_lt_u_imm16(result, lhs, rhs)
                }
                Instr::I32LeS { result, lhs, rhs } => self.execute_i32_le_s(result, lhs, rhs),
                Instr::I32LeSImm16 { result, lhs, rhs } => {
                    self.execute_i32_le_s_imm16(result, lhs, rhs)
                }
                Instr::I32LeU { result, lhs, rhs } => self.execute_i32_le_u(result, lhs, rhs),
                Instr::I32LeUImm16 { result, lhs, rhs } => {
                    self.execute_i32_le_u_imm16(result, lhs, rhs)
                }
                Instr::I32GtS { result, lhs, rhs } => self.execute_i32_gt_s(result, lhs, rhs),
                Instr::I32GtSImm16 { result, lhs, rhs } => {
                    self.execute_i32_gt_s_imm16(result, lhs, rhs)
                }
                Instr::I32GtU { result, lhs, rhs } => self.execute_i32_gt_u(result, lhs, rhs),
                Instr::I32GtUImm16 { result, lhs, rhs } => {
                    self.execute_i32_gt_u_imm16(result, lhs, rhs)
                }
                Instr::I32GeS { result, lhs, rhs } => self.execute_i32_ge_s(result, lhs, rhs),
                Instr::I32GeSImm16 { result, lhs, rhs } => {
                    self.execute_i32_ge_s_imm16(result, lhs, rhs)
                }
                Instr::I32GeU { result, lhs, rhs } => self.execute_i32_ge_u(result, lhs, rhs),
                Instr::I32GeUImm16 { result, lhs, rhs } => {
                    self.execute_i32_ge_u_imm16(result, lhs, rhs)
                }
                Instr::I64Eq { result, lhs, rhs } => self.execute_i64_eq(result, lhs, rhs),
                Instr::I64EqImm16 { result, lhs, rhs } => {
                    self.execute_i64_eq_imm16(result, lhs, rhs)
                }
                Instr::I64Ne { result, lhs, rhs } => self.execute_i64_ne(result, lhs, rhs),
                Instr::I64NeImm16 { result, lhs, rhs } => {
                    self.execute_i64_ne_imm16(result, lhs, rhs)
                }
                Instr::I64LtS { result, lhs, rhs } => self.execute_i64_lt_s(result, lhs, rhs),
                Instr::I64LtSImm16 { result, lhs, rhs } => {
                    self.execute_i64_lt_s_imm16(result, lhs, rhs)
                }
                Instr::I64LtU { result, lhs, rhs } => self.execute_i64_lt_u(result, lhs, rhs),
                Instr::I64LtUImm16 { result, lhs, rhs } => {
                    self.execute_i64_lt_u_imm16(result, lhs, rhs)
                }
                Instr::I64LeS { result, lhs, rhs } => self.execute_i64_le_s(result, lhs, rhs),
                Instr::I64LeSImm16 { result, lhs, rhs } => {
                    self.execute_i64_le_s_imm16(result, lhs, rhs)
                }
                Instr::I64LeU { result, lhs, rhs } => self.execute_i64_le_u(result, lhs, rhs),
                Instr::I64LeUImm16 { result, lhs, rhs } => {
                    self.execute_i64_le_u_imm16(result, lhs, rhs)
                }
                Instr::I64GtS { result, lhs, rhs } => self.execute_i64_gt_s(result, lhs, rhs),
                Instr::I64GtSImm16 { result, lhs, rhs } => {
                    self.execute_i64_gt_s_imm16(result, lhs, rhs)
                }
                Instr::I64GtU { result, lhs, rhs } => self.execute_i64_gt_u(result, lhs, rhs),
                Instr::I64GtUImm16 { result, lhs, rhs } => {
                    self.execute_i64_gt_u_imm16(result, lhs, rhs)
                }
                Instr::I64GeS { result, lhs, rhs } => self.execute_i64_ge_s(result, lhs, rhs),
                Instr::I64GeSImm16 { result, lhs, rhs } => {
                    self.execute_i64_ge_s_imm16(result, lhs, rhs)
                }
                Instr::I64GeU { result, lhs, rhs } => self.execute_i64_ge_u(result, lhs, rhs),
                Instr::I64GeUImm16 { result, lhs, rhs } => {
                    self.execute_i64_ge_u_imm16(result, lhs, rhs)
                }
                Instr::F32Eq { result, lhs, rhs } => self.execute_f32_eq(result, lhs, rhs),
                Instr::F32Ne { result, lhs, rhs } => self.execute_f32_ne(result, lhs, rhs),
                Instr::F32Lt { result, lhs, rhs } => self.execute_f32_lt(result, lhs, rhs),
                Instr::F32Le { result, lhs, rhs } => self.execute_f32_le(result, lhs, rhs),
                Instr::F32Gt { result, lhs, rhs } => self.execute_f32_gt(result, lhs, rhs),
                Instr::F32Ge { result, lhs, rhs } => self.execute_f32_ge(result, lhs, rhs),
                Instr::F64Eq { result, lhs, rhs } => self.execute_f64_eq(result, lhs, rhs),
                Instr::F64Ne { result, lhs, rhs } => self.execute_f64_ne(result, lhs, rhs),
                Instr::F64Lt { result, lhs, rhs } => self.execute_f64_lt(result, lhs, rhs),
                Instr::F64Le { result, lhs, rhs } => self.execute_f64_le(result, lhs, rhs),
                Instr::F64Gt { result, lhs, rhs } => self.execute_f64_gt(result, lhs, rhs),
                Instr::F64Ge { result, lhs, rhs } => self.execute_f64_ge(result, lhs, rhs),
                Instr::I32Clz { result, input } => self.execute_i32_clz(result, input),
                Instr::I32Ctz { result, input } => self.execute_i32_ctz(result, input),
                Instr::I32Popcnt { result, input } => self.execute_i32_popcnt(result, input),
                Instr::I32Add { result, lhs, rhs } => self.execute_i32_add(result, lhs, rhs),
                Instr::I32AddImm16 { result, lhs, rhs } => {
                    self.execute_i32_add_imm16(result, lhs, rhs)
                }
                Instr::I32Sub { result, lhs, rhs } => self.execute_i32_sub(result, lhs, rhs),
                Instr::I32SubImm16Lhs { result, lhs, rhs } => {
                    self.execute_i32_sub_imm16_lhs(result, lhs, rhs)
                }
                Instr::I32Mul { result, lhs, rhs } => self.execute_i32_mul(result, lhs, rhs),
                Instr::I32MulImm16 { result, lhs, rhs } => {
                    self.execute_i32_mul_imm16(result, lhs, rhs)
                }
                Instr::I32DivS { result, lhs, rhs } => self.execute_i32_div_s(result, lhs, rhs)?,
                Instr::I32DivSImm16Rhs { result, lhs, rhs } => {
                    self.execute_i32_div_s_imm16_rhs(result, lhs, rhs)?
                }
                Instr::I32DivSImm16Lhs { result, lhs, rhs } => {
                    self.execute_i32_div_s_imm16_lhs(result, lhs, rhs)?
                }
                Instr::I32DivU { result, lhs, rhs } => self.execute_i32_div_u(result, lhs, rhs)?,
                Instr::I32DivUImm16Rhs { result, lhs, rhs } => {
                    self.execute_i32_div_u_imm16_rhs(result, lhs, rhs)
                }
                Instr::I32DivUImm16Lhs { result, lhs, rhs } => {
                    self.execute_i32_div_u_imm16_lhs(result, lhs, rhs)?
                }
                Instr::I32RemS { result, lhs, rhs } => self.execute_i32_rem_s(result, lhs, rhs)?,
                Instr::I32RemSImm16Rhs { result, lhs, rhs } => {
                    self.execute_i32_rem_s_imm16_rhs(result, lhs, rhs)?
                }
                Instr::I32RemSImm16Lhs { result, lhs, rhs } => {
                    self.execute_i32_rem_s_imm16_lhs(result, lhs, rhs)?
                }
                Instr::I32RemU { result, lhs, rhs } => self.execute_i32_rem_u(result, lhs, rhs)?,
                Instr::I32RemUImm16Rhs { result, lhs, rhs } => {
                    self.execute_i32_rem_u_imm16_rhs(result, lhs, rhs)
                }
                Instr::I32RemUImm16Lhs { result, lhs, rhs } => {
                    self.execute_i32_rem_u_imm16_lhs(result, lhs, rhs)?
                }
                Instr::I32And { result, lhs, rhs } => self.execute_i32_and(result, lhs, rhs),
                Instr::I32AndEqz { result, lhs, rhs } => self.execute_i32_and_eqz(result, lhs, rhs),
                Instr::I32AndEqzImm16 { result, lhs, rhs } => {
                    self.execute_i32_and_eqz_imm16(result, lhs, rhs)
                }
                Instr::I32AndImm16 { result, lhs, rhs } => {
                    self.execute_i32_and_imm16(result, lhs, rhs)
                }
                Instr::I32Or { result, lhs, rhs } => self.execute_i32_or(result, lhs, rhs),
                Instr::I32OrEqz { result, lhs, rhs } => self.execute_i32_or_eqz(result, lhs, rhs),
                Instr::I32OrEqzImm16 { result, lhs, rhs } => {
                    self.execute_i32_or_eqz_imm16(result, lhs, rhs)
                }
                Instr::I32OrImm16 { result, lhs, rhs } => {
                    self.execute_i32_or_imm16(result, lhs, rhs)
                }
                Instr::I32Xor { result, lhs, rhs } => self.execute_i32_xor(result, lhs, rhs),
                Instr::I32XorEqz { result, lhs, rhs } => self.execute_i32_xor_eqz(result, lhs, rhs),
                Instr::I32XorEqzImm16 { result, lhs, rhs } => {
                    self.execute_i32_xor_eqz_imm16(result, lhs, rhs)
                }
                Instr::I32XorImm16 { result, lhs, rhs } => {
                    self.execute_i32_xor_imm16(result, lhs, rhs)
                }
                Instr::I32Shl { result, lhs, rhs } => self.execute_i32_shl(result, lhs, rhs),
                Instr::I32ShlBy { result, lhs, rhs } => self.execute_i32_shl_by(result, lhs, rhs),
                Instr::I32ShlImm16 { result, lhs, rhs } => {
                    self.execute_i32_shl_imm16(result, lhs, rhs)
                }
                Instr::I32ShrU { result, lhs, rhs } => self.execute_i32_shr_u(result, lhs, rhs),
                Instr::I32ShrUBy { result, lhs, rhs } => {
                    self.execute_i32_shr_u_by(result, lhs, rhs)
                }
                Instr::I32ShrUImm16 { result, lhs, rhs } => {
                    self.execute_i32_shr_u_imm16(result, lhs, rhs)
                }
                Instr::I32ShrS { result, lhs, rhs } => self.execute_i32_shr_s(result, lhs, rhs),
                Instr::I32ShrSBy { result, lhs, rhs } => {
                    self.execute_i32_shr_s_by(result, lhs, rhs)
                }
                Instr::I32ShrSImm16 { result, lhs, rhs } => {
                    self.execute_i32_shr_s_imm16(result, lhs, rhs)
                }
                Instr::I32Rotl { result, lhs, rhs } => self.execute_i32_rotl(result, lhs, rhs),
                Instr::I32RotlBy { result, lhs, rhs } => self.execute_i32_rotl_by(result, lhs, rhs),
                Instr::I32RotlImm16 { result, lhs, rhs } => {
                    self.execute_i32_rotl_imm16(result, lhs, rhs)
                }
                Instr::I32Rotr { result, lhs, rhs } => self.execute_i32_rotr(result, lhs, rhs),
                Instr::I32RotrBy { result, lhs, rhs } => self.execute_i32_rotr_by(result, lhs, rhs),
                Instr::I32RotrImm16 { result, lhs, rhs } => {
                    self.execute_i32_rotr_imm16(result, lhs, rhs)
                }
                Instr::I64Clz { result, input } => self.execute_i64_clz(result, input),
                Instr::I64Ctz { result, input } => self.execute_i64_ctz(result, input),
                Instr::I64Popcnt { result, input } => self.execute_i64_popcnt(result, input),
                Instr::I64Add { result, lhs, rhs } => self.execute_i64_add(result, lhs, rhs),
                Instr::I64AddImm16 { result, lhs, rhs } => {
                    self.execute_i64_add_imm16(result, lhs, rhs)
                }
                Instr::I64Sub { result, lhs, rhs } => self.execute_i64_sub(result, lhs, rhs),
                Instr::I64SubImm16Lhs { result, lhs, rhs } => {
                    self.execute_i64_sub_imm16_lhs(result, lhs, rhs)
                }
                Instr::I64Mul { result, lhs, rhs } => self.execute_i64_mul(result, lhs, rhs),
                Instr::I64MulImm16 { result, lhs, rhs } => {
                    self.execute_i64_mul_imm16(result, lhs, rhs)
                }
                Instr::I64DivS { result, lhs, rhs } => self.execute_i64_div_s(result, lhs, rhs)?,
                Instr::I64DivSImm16Rhs { result, lhs, rhs } => {
                    self.execute_i64_div_s_imm16_rhs(result, lhs, rhs)?
                }
                Instr::I64DivSImm16Lhs { result, lhs, rhs } => {
                    self.execute_i64_div_s_imm16_lhs(result, lhs, rhs)?
                }
                Instr::I64DivU { result, lhs, rhs } => self.execute_i64_div_u(result, lhs, rhs)?,
                Instr::I64DivUImm16Rhs { result, lhs, rhs } => {
                    self.execute_i64_div_u_imm16_rhs(result, lhs, rhs)
                }
                Instr::I64DivUImm16Lhs { result, lhs, rhs } => {
                    self.execute_i64_div_u_imm16_lhs(result, lhs, rhs)?
                }
                Instr::I64RemS { result, lhs, rhs } => self.execute_i64_rem_s(result, lhs, rhs)?,
                Instr::I64RemSImm16Rhs { result, lhs, rhs } => {
                    self.execute_i64_rem_s_imm16_rhs(result, lhs, rhs)?
                }
                Instr::I64RemSImm16Lhs { result, lhs, rhs } => {
                    self.execute_i64_rem_s_imm16_lhs(result, lhs, rhs)?
                }
                Instr::I64RemU { result, lhs, rhs } => self.execute_i64_rem_u(result, lhs, rhs)?,
                Instr::I64RemUImm16Rhs { result, lhs, rhs } => {
                    self.execute_i64_rem_u_imm16_rhs(result, lhs, rhs)
                }
                Instr::I64RemUImm16Lhs { result, lhs, rhs } => {
                    self.execute_i64_rem_u_imm16_lhs(result, lhs, rhs)?
                }
                Instr::I64And { result, lhs, rhs } => self.execute_i64_and(result, lhs, rhs),
                Instr::I64AndImm16 { result, lhs, rhs } => {
                    self.execute_i64_and_imm16(result, lhs, rhs)
                }
                Instr::I64Or { result, lhs, rhs } => self.execute_i64_or(result, lhs, rhs),
                Instr::I64OrImm16 { result, lhs, rhs } => {
                    self.execute_i64_or_imm16(result, lhs, rhs)
                }
                Instr::I64Xor { result, lhs, rhs } => self.execute_i64_xor(result, lhs, rhs),
                Instr::I64XorImm16 { result, lhs, rhs } => {
                    self.execute_i64_xor_imm16(result, lhs, rhs)
                }
                Instr::I64Shl { result, lhs, rhs } => self.execute_i64_shl(result, lhs, rhs),
                Instr::I64ShlBy { result, lhs, rhs } => self.execute_i64_shl_by(result, lhs, rhs),
                Instr::I64ShlImm16 { result, lhs, rhs } => {
                    self.execute_i64_shl_imm16(result, lhs, rhs)
                }
                Instr::I64ShrU { result, lhs, rhs } => self.execute_i64_shr_u(result, lhs, rhs),
                Instr::I64ShrUBy { result, lhs, rhs } => {
                    self.execute_i64_shr_u_by(result, lhs, rhs)
                }
                Instr::I64ShrUImm16 { result, lhs, rhs } => {
                    self.execute_i64_shr_u_imm16(result, lhs, rhs)
                }
                Instr::I64ShrS { result, lhs, rhs } => self.execute_i64_shr_s(result, lhs, rhs),
                Instr::I64ShrSBy { result, lhs, rhs } => {
                    self.execute_i64_shr_s_by(result, lhs, rhs)
                }
                Instr::I64ShrSImm16 { result, lhs, rhs } => {
                    self.execute_i64_shr_s_imm16(result, lhs, rhs)
                }
                Instr::I64Rotl { result, lhs, rhs } => self.execute_i64_rotl(result, lhs, rhs),
                Instr::I64RotlBy { result, lhs, rhs } => self.execute_i64_rotl_by(result, lhs, rhs),
                Instr::I64RotlImm16 { result, lhs, rhs } => {
                    self.execute_i64_rotl_imm16(result, lhs, rhs)
                }
                Instr::I64Rotr { result, lhs, rhs } => self.execute_i64_rotr(result, lhs, rhs),
                Instr::I64RotrBy { result, lhs, rhs } => self.execute_i64_rotr_by(result, lhs, rhs),
                Instr::I64RotrImm16 { result, lhs, rhs } => {
                    self.execute_i64_rotr_imm16(result, lhs, rhs)
                }
                Instr::I32WrapI64 { result, input } => self.execute_i32_wrap_i64(result, input),
                Instr::I32Extend8S { result, input } => self.execute_i32_extend8_s(result, input),
                Instr::I32Extend16S { result, input } => self.execute_i32_extend16_s(result, input),
                Instr::I64Extend8S { result, input } => self.execute_i64_extend8_s(result, input),
                Instr::I64Extend16S { result, input } => self.execute_i64_extend16_s(result, input),
                Instr::I64Extend32S { result, input } => self.execute_i64_extend32_s(result, input),
                Instr::F32Abs { result, input } => self.execute_f32_abs(result, input),
                Instr::F32Neg { result, input } => self.execute_f32_neg(result, input),
                Instr::F32Ceil { result, input } => self.execute_f32_ceil(result, input),
                Instr::F32Floor { result, input } => self.execute_f32_floor(result, input),
                Instr::F32Trunc { result, input } => self.execute_f32_trunc(result, input),
                Instr::F32Nearest { result, input } => self.execute_f32_nearest(result, input),
                Instr::F32Sqrt { result, input } => self.execute_f32_sqrt(result, input),
                Instr::F32Add { result, lhs, rhs } => self.execute_f32_add(result, lhs, rhs),
                Instr::F32Sub { result, lhs, rhs } => self.execute_f32_sub(result, lhs, rhs),
                Instr::F32Mul { result, lhs, rhs } => self.execute_f32_mul(result, lhs, rhs),
                Instr::F32Div { result, lhs, rhs } => self.execute_f32_div(result, lhs, rhs),
                Instr::F32Min { result, lhs, rhs } => self.execute_f32_min(result, lhs, rhs),
                Instr::F32Max { result, lhs, rhs } => self.execute_f32_max(result, lhs, rhs),
                Instr::F32Copysign { result, lhs, rhs } => {
                    self.execute_f32_copysign(result, lhs, rhs)
                }
                Instr::F32CopysignImm { result, lhs, rhs } => {
                    self.execute_f32_copysign_imm(result, lhs, rhs)
                }
                Instr::F64Abs { result, input } => self.execute_f64_abs(result, input),
                Instr::F64Neg { result, input } => self.execute_f64_neg(result, input),
                Instr::F64Ceil { result, input } => self.execute_f64_ceil(result, input),
                Instr::F64Floor { result, input } => self.execute_f64_floor(result, input),
                Instr::F64Trunc { result, input } => self.execute_f64_trunc(result, input),
                Instr::F64Nearest { result, input } => self.execute_f64_nearest(result, input),
                Instr::F64Sqrt { result, input } => self.execute_f64_sqrt(result, input),
                Instr::F64Add { result, lhs, rhs } => self.execute_f64_add(result, lhs, rhs),
                Instr::F64Sub { result, lhs, rhs } => self.execute_f64_sub(result, lhs, rhs),
                Instr::F64Mul { result, lhs, rhs } => self.execute_f64_mul(result, lhs, rhs),
                Instr::F64Div { result, lhs, rhs } => self.execute_f64_div(result, lhs, rhs),
                Instr::F64Min { result, lhs, rhs } => self.execute_f64_min(result, lhs, rhs),
                Instr::F64Max { result, lhs, rhs } => self.execute_f64_max(result, lhs, rhs),
                Instr::F64Copysign { result, lhs, rhs } => {
                    self.execute_f64_copysign(result, lhs, rhs)
                }
                Instr::F64CopysignImm { result, lhs, rhs } => {
                    self.execute_f64_copysign_imm(result, lhs, rhs)
                }
                Instr::I32TruncF32S { result, input } => {
                    self.execute_i32_trunc_f32_s(result, input)?
                }
                Instr::I32TruncF32U { result, input } => {
                    self.execute_i32_trunc_f32_u(result, input)?
                }
                Instr::I32TruncF64S { result, input } => {
                    self.execute_i32_trunc_f64_s(result, input)?
                }
                Instr::I32TruncF64U { result, input } => {
                    self.execute_i32_trunc_f64_u(result, input)?
                }
                Instr::I64TruncF32S { result, input } => {
                    self.execute_i64_trunc_f32_s(result, input)?
                }
                Instr::I64TruncF32U { result, input } => {
                    self.execute_i64_trunc_f32_u(result, input)?
                }
                Instr::I64TruncF64S { result, input } => {
                    self.execute_i64_trunc_f64_s(result, input)?
                }
                Instr::I64TruncF64U { result, input } => {
                    self.execute_i64_trunc_f64_u(result, input)?
                }
                Instr::I32TruncSatF32S { result, input } => {
                    self.execute_i32_trunc_sat_f32_s(result, input)
                }
                Instr::I32TruncSatF32U { result, input } => {
                    self.execute_i32_trunc_sat_f32_u(result, input)
                }
                Instr::I32TruncSatF64S { result, input } => {
                    self.execute_i32_trunc_sat_f64_s(result, input)
                }
                Instr::I32TruncSatF64U { result, input } => {
                    self.execute_i32_trunc_sat_f64_u(result, input)
                }
                Instr::I64TruncSatF32S { result, input } => {
                    self.execute_i64_trunc_sat_f32_s(result, input)
                }
                Instr::I64TruncSatF32U { result, input } => {
                    self.execute_i64_trunc_sat_f32_u(result, input)
                }
                Instr::I64TruncSatF64S { result, input } => {
                    self.execute_i64_trunc_sat_f64_s(result, input)
                }
                Instr::I64TruncSatF64U { result, input } => {
                    self.execute_i64_trunc_sat_f64_u(result, input)
                }
                Instr::F32DemoteF64 { result, input } => self.execute_f32_demote_f64(result, input),
                Instr::F64PromoteF32 { result, input } => {
                    self.execute_f64_promote_f32(result, input)
                }
                Instr::F32ConvertI32S { result, input } => {
                    self.execute_f32_convert_i32_s(result, input)
                }
                Instr::F32ConvertI32U { result, input } => {
                    self.execute_f32_convert_i32_u(result, input)
                }
                Instr::F32ConvertI64S { result, input } => {
                    self.execute_f32_convert_i64_s(result, input)
                }
                Instr::F32ConvertI64U { result, input } => {
                    self.execute_f32_convert_i64_u(result, input)
                }
                Instr::F64ConvertI32S { result, input } => {
                    self.execute_f64_convert_i32_s(result, input)
                }
                Instr::F64ConvertI32U { result, input } => {
                    self.execute_f64_convert_i32_u(result, input)
                }
                Instr::F64ConvertI64S { result, input } => {
                    self.execute_f64_convert_i64_s(result, input)
                }
                Instr::F64ConvertI64U { result, input } => {
                    self.execute_f64_convert_i64_u(result, input)
                }
                Instr::TableGet { result, index } => {
                    self.execute_table_get(&store.inner, result, index)?
                }
                Instr::TableGetImm { result, index } => {
                    self.execute_table_get_imm(&store.inner, result, index)?
                }
                Instr::TableSize { result, table } => {
                    self.execute_table_size(&store.inner, result, table)
                }
                Instr::TableSet { index, value } => {
                    self.execute_table_set(&mut store.inner, index, value)?
                }
                Instr::TableSetAt { index, value } => {
                    self.execute_table_set_at(&mut store.inner, index, value)?
                }
                Instr::TableCopy { dst, src, len } => {
                    self.execute_table_copy(&mut store.inner, dst, src, len)?
                }
                Instr::TableCopyTo { dst, src, len } => {
                    self.execute_table_copy_to(&mut store.inner, dst, src, len)?
                }
                Instr::TableCopyFrom { dst, src, len } => {
                    self.execute_table_copy_from(&mut store.inner, dst, src, len)?
                }
                Instr::TableCopyFromTo { dst, src, len } => {
                    self.execute_table_copy_from_to(&mut store.inner, dst, src, len)?
                }
                Instr::TableCopyExact { dst, src, len } => {
                    self.execute_table_copy_exact(&mut store.inner, dst, src, len)?
                }
                Instr::TableCopyToExact { dst, src, len } => {
                    self.execute_table_copy_to_exact(&mut store.inner, dst, src, len)?
                }
                Instr::TableCopyFromExact { dst, src, len } => {
                    self.execute_table_copy_from_exact(&mut store.inner, dst, src, len)?
                }
                Instr::TableCopyFromToExact { dst, src, len } => {
                    self.execute_table_copy_from_to_exact(&mut store.inner, dst, src, len)?
                }
                Instr::TableInit { dst, src, len } => {
                    self.execute_table_init(&mut store.inner, dst, src, len)?
                }
                Instr::TableInitTo { dst, src, len } => {
                    self.execute_table_init_to(&mut store.inner, dst, src, len)?
                }
                Instr::TableInitFrom { dst, src, len } => {
                    self.execute_table_init_from(&mut store.inner, dst, src, len)?
                }
                Instr::TableInitFromTo { dst, src, len } => {
                    self.execute_table_init_from_to(&mut store.inner, dst, src, len)?
                }
                Instr::TableInitExact { dst, src, len } => {
                    self.execute_table_init_exact(&mut store.inner, dst, src, len)?
                }
                Instr::TableInitToExact { dst, src, len } => {
                    self.execute_table_init_to_exact(&mut store.inner, dst, src, len)?
                }
                Instr::TableInitFromExact { dst, src, len } => {
                    self.execute_table_init_from_exact(&mut store.inner, dst, src, len)?
                }
                Instr::TableInitFromToExact { dst, src, len } => {
                    self.execute_table_init_from_to_exact(&mut store.inner, dst, src, len)?
                }
                Instr::TableFill { dst, len, value } => {
                    self.execute_table_fill(&mut store.inner, dst, len, value)?
                }
                Instr::TableFillAt { dst, len, value } => {
                    self.execute_table_fill_at(&mut store.inner, dst, len, value)?
                }
                Instr::TableFillExact { dst, len, value } => {
                    self.execute_table_fill_exact(&mut store.inner, dst, len, value)?
                }
                Instr::TableFillAtExact { dst, len, value } => {
                    self.execute_table_fill_at_exact(&mut store.inner, dst, len, value)?
                }
                Instr::TableGrow {
                    result,
                    delta,
                    value,
                } => self.execute_table_grow(store, result, delta, value)?,
                Instr::TableGrowImm {
                    result,
                    delta,
                    value,
                } => self.execute_table_grow_imm(store, result, delta, value)?,
                Instr::ElemDrop { index } => self.execute_element_drop(&mut store.inner, index),
                Instr::DataDrop { index } => self.execute_data_drop(&mut store.inner, index),
                Instr::MemorySize { result, memory } => {
                    self.execute_memory_size(&store.inner, result, memory)
                }
                Instr::MemoryGrow { result, delta } => {
                    self.execute_memory_grow(store, result, delta)?
                }
                Instr::MemoryGrowBy { result, delta } => {
                    self.execute_memory_grow_by(store, result, delta)?
                }
                Instr::MemoryCopy { dst, src, len } => {
                    self.execute_memory_copy(&mut store.inner, dst, src, len)?
                }
                Instr::MemoryCopyTo { dst, src, len } => {
                    self.execute_memory_copy_to(&mut store.inner, dst, src, len)?
                }
                Instr::MemoryCopyFrom { dst, src, len } => {
                    self.execute_memory_copy_from(&mut store.inner, dst, src, len)?
                }
                Instr::MemoryCopyFromTo { dst, src, len } => {
                    self.execute_memory_copy_from_to(&mut store.inner, dst, src, len)?
                }
                Instr::MemoryCopyExact { dst, src, len } => {
                    self.execute_memory_copy_exact(&mut store.inner, dst, src, len)?
                }
                Instr::MemoryCopyToExact { dst, src, len } => {
                    self.execute_memory_copy_to_exact(&mut store.inner, dst, src, len)?
                }
                Instr::MemoryCopyFromExact { dst, src, len } => {
                    self.execute_memory_copy_from_exact(&mut store.inner, dst, src, len)?
                }
                Instr::MemoryCopyFromToExact { dst, src, len } => {
                    self.execute_memory_copy_from_to_exact(&mut store.inner, dst, src, len)?
                }
                Instr::MemoryFill { dst, value, len } => {
                    self.execute_memory_fill(&mut store.inner, dst, value, len)?
                }
                Instr::MemoryFillAt { dst, value, len } => {
                    self.execute_memory_fill_at(&mut store.inner, dst, value, len)?
                }
                Instr::MemoryFillImm { dst, value, len } => {
                    self.execute_memory_fill_imm(&mut store.inner, dst, value, len)?
                }
                Instr::MemoryFillExact { dst, value, len } => {
                    self.execute_memory_fill_exact(&mut store.inner, dst, value, len)?
                }
                Instr::MemoryFillAtImm { dst, value, len } => {
                    self.execute_memory_fill_at_imm(&mut store.inner, dst, value, len)?
                }
                Instr::MemoryFillAtExact { dst, value, len } => {
                    self.execute_memory_fill_at_exact(&mut store.inner, dst, value, len)?
                }
                Instr::MemoryFillImmExact { dst, value, len } => {
                    self.execute_memory_fill_imm_exact(&mut store.inner, dst, value, len)?
                }
                Instr::MemoryFillAtImmExact { dst, value, len } => {
                    self.execute_memory_fill_at_imm_exact(&mut store.inner, dst, value, len)?
                }
                Instr::MemoryInit { dst, src, len } => {
                    self.execute_memory_init(&mut store.inner, dst, src, len)?
                }
                Instr::MemoryInitTo { dst, src, len } => {
                    self.execute_memory_init_to(&mut store.inner, dst, src, len)?
                }
                Instr::MemoryInitFrom { dst, src, len } => {
                    self.execute_memory_init_from(&mut store.inner, dst, src, len)?
                }
                Instr::MemoryInitFromTo { dst, src, len } => {
                    self.execute_memory_init_from_to(&mut store.inner, dst, src, len)?
                }
                Instr::MemoryInitExact { dst, src, len } => {
                    self.execute_memory_init_exact(&mut store.inner, dst, src, len)?
                }
                Instr::MemoryInitToExact { dst, src, len } => {
                    self.execute_memory_init_to_exact(&mut store.inner, dst, src, len)?
                }
                Instr::MemoryInitFromExact { dst, src, len } => {
                    self.execute_memory_init_from_exact(&mut store.inner, dst, src, len)?
                }
                Instr::MemoryInitFromToExact { dst, src, len } => {
                    self.execute_memory_init_from_to_exact(&mut store.inner, dst, src, len)?
                }
                Instr::TableIndex { .. }
                | Instr::MemoryIndex { .. }
                | Instr::DataIndex { .. }
                | Instr::ElemIndex { .. }
                | Instr::Const32 { .. }
                | Instr::I64Const32 { .. }
                | Instr::F64Const32 { .. }
                | Instr::BranchTableTarget { .. }
                | Instr::BranchTableTargetNonOverlapping { .. }
                | Instr::Register { .. }
                | Instr::Register2 { .. }
                | Instr::Register3 { .. }
                | Instr::RegisterAndImm32 { .. }
                | Instr::Imm16AndImm32 { .. }
                | Instr::RegisterSpan { .. }
                | Instr::RegisterList { .. }
                | Instr::CallIndirectParams { .. }
                | Instr::CallIndirectParamsImm16 { .. } => self.invalid_instruction_word()?,
            }
        }
    }
}

macro_rules! get_entity {
    (
        $(
            fn $name:ident(&self, index: $index_ty:ty) -> $id_ty:ty;
        )*
    ) => {
        $(
            #[doc = ::core::concat!(
                "Returns the [`",
                ::core::stringify!($id_ty),
                "`] at `index` for the currently used [`Instance`].\n\n",
                "# Panics\n\n",
                "- If there is no [`",
                ::core::stringify!($id_ty),
                "`] at `index` for the currently used [`Instance`] in `store`."
            )]
            #[inline]
            fn $name(&self, index: $index_ty) -> $id_ty {
                unsafe { self.cache.$name(index) }
                    .unwrap_or_else(|| {
                        const ENTITY_NAME: &'static str = ::core::stringify!($id_ty);
                        // Safety: within the Wasmi executor it is assumed that store entity
                        //         indices within the Wasmi bytecode are always valid for the
                        //         store. This is an invariant of the Wasmi translation.
                        unsafe {
                            unreachable_unchecked!(
                                "missing {ENTITY_NAME} at index {index:?} for the currently used instance",
                            )
                        }
                    })
            }
        )*
    }
}

impl Executor<'_> {
    get_entity! {
        fn get_func(&self, index: index::Func) -> Func;
        fn get_func_type_dedup(&self, index: index::FuncType) -> DedupFuncType;
        fn get_memory(&self, index: index::Memory) -> Memory;
        fn get_table(&self, index: index::Table) -> Table;
        fn get_global(&self, index: index::Global) -> Global;
        fn get_data_segment(&self, index: index::Data) -> DataSegment;
        fn get_element_segment(&self, index: index::Elem) -> ElementSegment;
    }

    /// Returns the [`Reg`] value.
    fn get_register(&self, register: Reg) -> UntypedVal {
        // Safety: - It is the responsibility of the `Executor`
        //           implementation to keep the `sp` pointer valid
        //           whenever this method is accessed.
        //         - This is done by updating the `sp` pointer whenever
        //           the heap underlying the value stack is changed.
        unsafe { self.sp.get(register) }
    }

    /// Returns the [`Reg`] value.
    fn get_register_as<T>(&self, register: Reg) -> T
    where
        T: From<UntypedVal>,
    {
        T::from(self.get_register(register))
    }

    /// Sets the [`Reg`] value to `value`.
    fn set_register(&mut self, register: Reg, value: impl Into<UntypedVal>) {
        // Safety: - It is the responsibility of the `Executor`
        //           implementation to keep the `sp` pointer valid
        //           whenever this method is accessed.
        //         - This is done by updating the `sp` pointer whenever
        //           the heap underlying the value stack is changed.
        unsafe { self.sp.set(register, value.into()) };
    }

    /// Shifts the instruction pointer to the next instruction.
    #[inline(always)]
    fn next_instr(&mut self) {
        self.next_instr_at(1)
    }

    /// Shifts the instruction pointer to the next instruction.
    ///
    /// Has a parameter `skip` to denote how many instruction words
    /// to skip to reach the next actual instruction.
    ///
    /// # Note
    ///
    /// This is used by Wasmi instructions that have a fixed
    /// encoding size of two instruction words such as [`Instruction::Branch`].
    #[inline(always)]
    fn next_instr_at(&mut self, skip: usize) {
        self.ip.add(skip)
    }

    /// Shifts the instruction pointer to the next instruction and returns `Ok(())`.
    ///
    /// # Note
    ///
    /// This is a convenience function for fallible instructions.
    #[inline(always)]
    fn try_next_instr(&mut self) -> Result<(), Error> {
        self.try_next_instr_at(1)
    }

    /// Shifts the instruction pointer to the next instruction and returns `Ok(())`.
    ///
    /// Has a parameter `skip` to denote how many instruction words
    /// to skip to reach the next actual instruction.
    ///
    /// # Note
    ///
    /// This is a convenience function for fallible instructions.
    #[inline(always)]
    fn try_next_instr_at(&mut self, skip: usize) -> Result<(), Error> {
        self.next_instr_at(skip);
        Ok(())
    }

    /// Returns the [`FrameRegisters`] of the [`CallFrame`].
    fn frame_stack_ptr_impl(value_stack: &mut ValueStack, frame: &CallFrame) -> FrameRegisters {
        // Safety: We are using the frame's own base offset as input because it is
        //         guaranteed by the Wasm validation and translation phase to be
        //         valid for all register indices used by the associated function body.
        unsafe { value_stack.stack_ptr_at(frame.base_offset()) }
    }

    /// Initializes the [`Executor`] state for the [`CallFrame`].
    ///
    /// # Note
    ///
    /// The initialization of the [`Executor`] allows for efficient execution.
    fn init_call_frame(&mut self, frame: &CallFrame) {
        Self::init_call_frame_impl(&mut self.stack.values, &mut self.sp, &mut self.ip, frame)
    }

    /// Initializes the [`Executor`] state for the [`CallFrame`].
    ///
    /// # Note
    ///
    /// The initialization of the [`Executor`] allows for efficient execution.
    fn init_call_frame_impl(
        value_stack: &mut ValueStack,
        sp: &mut FrameRegisters,
        ip: &mut InstructionPtr,
        frame: &CallFrame,
    ) {
        *sp = Self::frame_stack_ptr_impl(value_stack, frame);
        *ip = frame.instr_ptr();
    }

    /// Executes a generic unary [`Instruction`].
    #[inline(always)]
    fn execute_unary(&mut self, result: Reg, input: Reg, op: fn(UntypedVal) -> UntypedVal) {
        let value = self.get_register(input);
        self.set_register(result, op(value));
        self.next_instr();
    }

    /// Executes a fallible generic unary [`Instruction`].
    #[inline(always)]
    fn try_execute_unary(
        &mut self,
        result: Reg,
        input: Reg,
        op: fn(UntypedVal) -> Result<UntypedVal, TrapCode>,
    ) -> Result<(), Error> {
        let value = self.get_register(input);
        self.set_register(result, op(value)?);
        self.try_next_instr()
    }

    /// Executes a generic binary [`Instruction`].
    #[inline(always)]
    fn execute_binary(
        &mut self,
        result: Reg,
        lhs: Reg,
        rhs: Reg,
        op: fn(UntypedVal, UntypedVal) -> UntypedVal,
    ) {
        let lhs = self.get_register(lhs);
        let rhs = self.get_register(rhs);
        self.set_register(result, op(lhs, rhs));
        self.next_instr();
    }

    /// Executes a generic binary [`Instruction`].
    #[inline(always)]
    fn execute_binary_imm16<T>(
        &mut self,
        result: Reg,
        lhs: Reg,
        rhs: Const16<T>,
        op: fn(UntypedVal, UntypedVal) -> UntypedVal,
    ) where
        T: From<Const16<T>>,
        UntypedVal: From<T>,
    {
        let lhs = self.get_register(lhs);
        let rhs = UntypedVal::from(<T>::from(rhs));
        self.set_register(result, op(lhs, rhs));
        self.next_instr();
    }

    /// Executes a generic binary [`Instruction`] with reversed operands.
    #[inline(always)]
    fn execute_binary_imm16_lhs<T>(
        &mut self,
        result: Reg,
        lhs: Const16<T>,
        rhs: Reg,
        op: fn(UntypedVal, UntypedVal) -> UntypedVal,
    ) where
        T: From<Const16<T>>,
        UntypedVal: From<T>,
    {
        let lhs = UntypedVal::from(<T>::from(lhs));
        let rhs = self.get_register(rhs);
        self.set_register(result, op(lhs, rhs));
        self.next_instr();
    }

    /// Executes a generic shift or rotate [`Instruction`].
    #[inline(always)]
    fn execute_shift_by<T>(
        &mut self,
        result: Reg,
        lhs: Reg,
        rhs: ShiftAmount<T>,
        op: fn(UntypedVal, UntypedVal) -> UntypedVal,
    ) where
        T: From<ShiftAmount<T>>,
        UntypedVal: From<T>,
    {
        let lhs = self.get_register(lhs);
        let rhs = UntypedVal::from(<T>::from(rhs));
        self.set_register(result, op(lhs, rhs));
        self.next_instr();
    }

    /// Executes a fallible generic binary [`Instruction`].
    #[inline(always)]
    fn try_execute_binary(
        &mut self,
        result: Reg,
        lhs: Reg,
        rhs: Reg,
        op: fn(UntypedVal, UntypedVal) -> Result<UntypedVal, TrapCode>,
    ) -> Result<(), Error> {
        let lhs = self.get_register(lhs);
        let rhs = self.get_register(rhs);
        self.set_register(result, op(lhs, rhs)?);
        self.try_next_instr()
    }

    /// Executes a fallible generic binary [`Instruction`].
    #[inline(always)]
    fn try_execute_divrem_imm16_rhs<NonZeroT>(
        &mut self,
        result: Reg,
        lhs: Reg,
        rhs: Const16<NonZeroT>,
        op: fn(UntypedVal, NonZeroT) -> Result<UntypedVal, Error>,
    ) -> Result<(), Error>
    where
        NonZeroT: From<Const16<NonZeroT>>,
    {
        let lhs = self.get_register(lhs);
        let rhs = <NonZeroT>::from(rhs);
        self.set_register(result, op(lhs, rhs)?);
        self.try_next_instr()
    }

    /// Executes a fallible generic binary [`Instruction`].
    #[inline(always)]
    fn execute_divrem_imm16_rhs<NonZeroT>(
        &mut self,
        result: Reg,
        lhs: Reg,
        rhs: Const16<NonZeroT>,
        op: fn(UntypedVal, NonZeroT) -> UntypedVal,
    ) where
        NonZeroT: From<Const16<NonZeroT>>,
    {
        let lhs = self.get_register(lhs);
        let rhs = <NonZeroT>::from(rhs);
        self.set_register(result, op(lhs, rhs));
        self.next_instr()
    }

    /// Executes a fallible generic binary [`Instruction`] with reversed operands.
    #[inline(always)]
    fn try_execute_binary_imm16_lhs<T>(
        &mut self,
        result: Reg,
        lhs: Const16<T>,
        rhs: Reg,
        op: fn(UntypedVal, UntypedVal) -> Result<UntypedVal, TrapCode>,
    ) -> Result<(), Error>
    where
        T: From<Const16<T>>,
        UntypedVal: From<T>,
    {
        let lhs = UntypedVal::from(<T>::from(lhs));
        let rhs = self.get_register(rhs);
        self.set_register(result, op(lhs, rhs)?);
        self.try_next_instr()
    }

    /// Skips all [`Instruction`]s belonging to an [`Instruction::RegisterList`] encoding.
    #[inline(always)]
    fn skip_register_list(ip: InstructionPtr) -> InstructionPtr {
        let mut ip = ip;
        while let Instruction::RegisterList { .. } = *ip.get() {
            ip.add(1);
        }
        // We skip an additional `Instruction` because we know that `Instruction::RegisterList` is always followed by one of:
        // - `Instruction::Register`
        // - `Instruction::Register2`
        // - `Instruction::Register3`.
        ip.add(1);
        ip
    }

    /// Returns the optional `memory` parameter for a `load_at` [`Instruction`].
    ///
    /// # Note
    ///
    /// - Returns the default [`index::Memory`] if the parameter is missing.
    /// - Bumps `self.ip` if a [`Instruction::MemoryIndex`] parameter was found.
    #[inline(always)]
    fn fetch_optional_memory(&mut self) -> index::Memory {
        let mut addr: InstructionPtr = self.ip;
        addr.add(1);
        match *addr.get() {
            Instruction::MemoryIndex { index } => {
                hint::cold();
                self.ip = addr;
                index
            }
            _ => index::Memory::from(0),
        }
    }
}

impl Executor<'_> {
    /// Used for all [`Instruction`] words that are not meant for execution.
    ///
    /// # Note
    ///
    /// This includes [`Instruction`] variants such as [`Instruction::TableIndex`]
    /// that primarily carry parameters for actually executable [`Instruction`].
    fn invalid_instruction_word(&mut self) -> Result<(), Error> {
        // Safety: Wasmi translation guarantees that branches are never taken to instruction parameters directly.
        unsafe {
            unreachable_unchecked!(
                "expected instruction but found instruction parameter: {:?}",
                *self.ip.get()
            )
        }
    }

    /// Executes a Wasm `unreachable` instruction.
    fn execute_trap(&mut self, trap_code: TrapCode) -> Result<(), Error> {
        Err(Error::from(trap_code))
    }

    /// Executes an [`Instruction::ConsumeFuel`].
    fn execute_consume_fuel(
        &mut self,
        store: &mut StoreInner,
        block_fuel: BlockFuel,
    ) -> Result<(), Error> {
        // We do not have to check if fuel metering is enabled since
        // [`Instruction::ConsumeFuel`] are only generated if fuel metering
        // is enabled to begin with.
        store
            .fuel_mut()
            .consume_fuel_unchecked(block_fuel.to_u64())?;
        self.try_next_instr()
    }

    /// Executes an [`Instruction::RefFunc`].
    fn execute_ref_func(&mut self, result: Reg, func_index: index::Func) {
        let func = self.get_func(func_index);
        let funcref = FuncRef::new(func);
        self.set_register(result, funcref);
        self.next_instr();
    }
}

/// Extension method for [`UntypedVal`] required by the [`Executor`].
trait UntypedValueExt {
    /// Executes a fused `i32.and` + `i32.eqz` instruction.
    fn i32_and_eqz(x: UntypedVal, y: UntypedVal) -> UntypedVal;

    /// Executes a fused `i32.or` + `i32.eqz` instruction.
    fn i32_or_eqz(x: UntypedVal, y: UntypedVal) -> UntypedVal;

    /// Executes a fused `i32.xor` + `i32.eqz` instruction.
    fn i32_xor_eqz(x: UntypedVal, y: UntypedVal) -> UntypedVal;
}

impl UntypedValueExt for UntypedVal {
    fn i32_and_eqz(x: UntypedVal, y: UntypedVal) -> UntypedVal {
        (i32::from(UntypedVal::i32_and(x, y)) == 0).into()
    }

    fn i32_or_eqz(x: UntypedVal, y: UntypedVal) -> UntypedVal {
        (i32::from(UntypedVal::i32_or(x, y)) == 0).into()
    }

    fn i32_xor_eqz(x: UntypedVal, y: UntypedVal) -> UntypedVal {
        (i32::from(UntypedVal::i32_xor(x, y)) == 0).into()
    }
}