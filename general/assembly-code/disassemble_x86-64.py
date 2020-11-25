# We need to emulate ARM and x86 code
from unicorn import Uc, UC_ARCH_X86, UC_MODE_64, UcError
# for accessing the RAX and RDI registers
from unicorn.x86_const import UC_X86_REG_RDI, UC_X86_REG_RAX
# We need to disassemble x86_64 code
from capstone import Cs, CS_ARCH_X86, CS_MODE_64, CsError


X86_MACHINE_CODE = b"\x48\x31\xc0\x48\xff\xc0\x48\x85\xff\x0f\x84\x0d\x00\x00\x00\x48\x99\x48\xf7\xe7\x48\xff\xcf\xe9\xea\xff\xff\xff"

# memory address where emulation starts
ADDRESS = 0x1000000

try:
      # Initialize the disassembler in x86 mode
      md = Cs(CS_ARCH_X86, CS_MODE_64)
      # iterate over each instruction and print it
      for instruction in md.disasm(X86_MACHINE_CODE, 0x1000):
            print("0x%x:\t%s\t%s" % (instruction.address, instruction.mnemonic, instruction.op_str))
except CsError as e:
      print("Capstone Error: %s" % e)

try:
    # Initialize emulator in x86_64 mode
    mu = Uc(UC_ARCH_X86, UC_MODE_64)
    # map 2MB memory for this emulation
    mu.mem_map(ADDRESS, 2 * 1024 * 1024)
    # write machine code to be emulated to memory
    mu.mem_write(ADDRESS, X86_MACHINE_CODE)
    # Set the r0 register in the code to the number of 7
    mu.reg_write(UC_X86_REG_RDI, 7)
    # emulate code in infinite time & unlimited instructions
    mu.emu_start(ADDRESS, ADDRESS + len(X86_MACHINE_CODE))
    # now print out the R0 register
    print("Emulation done. Below is the result")
    rax = mu.reg_read(UC_X86_REG_RAX)
    print(">>> RAX = %u" % rax)
except UcError as e:
    print("Unicorn Error: %s" % e)