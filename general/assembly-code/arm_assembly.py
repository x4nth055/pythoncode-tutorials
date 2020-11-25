# We need to emulate ARM
from unicorn import Uc, UC_ARCH_ARM, UC_MODE_ARM, UcError
# for accessing the R0 and R1 registers
from unicorn.arm_const import UC_ARM_REG_R0, UC_ARM_REG_R1
# We need to assemble ARM code
from keystone import Ks, KS_ARCH_ARM, KS_MODE_ARM, KsError


ARM_CODE = """
// n is r0, we will pass it from python, ans is r1
mov r1, 1       	// ans = 1
loop:
cmp r0, 0       	// while n >= 0:
mulgt r1, r1, r0	//   ans *= n
subgt r0, r0, 1 	//   n = n - 1
bgt loop        	// 
                	// answer is in r1
"""

print("Assembling the ARM code")
try:
    # initialize the keystone object with the ARM architecture
    ks = Ks(KS_ARCH_ARM, KS_MODE_ARM)
    # Assemble the ARM code
    ARM_BYTECODE, _ = ks.asm(ARM_CODE)
	# convert the array of integers into bytes
    ARM_BYTECODE = bytes(ARM_BYTECODE)
    print(f"Code successfully assembled (length = {len(ARM_BYTECODE)})")
    print("ARM bytecode:", ARM_BYTECODE)
except KsError as e:
    print("Keystone Error: %s" % e)
    exit(1)


# memory address where emulation starts
ADDRESS = 0x1000000

print("Emulating the ARM code")
try:
    # Initialize emulator in ARM mode
    mu = Uc(UC_ARCH_ARM, UC_MODE_ARM)
    # map 2MB memory for this emulation
    mu.mem_map(ADDRESS, 2 * 1024 * 1024)
    # write machine code to be emulated to memory
    mu.mem_write(ADDRESS, ARM_BYTECODE)
    # Set the r0 register in the code, let's calculate factorial(5)
    mu.reg_write(UC_ARM_REG_R0, 5)
    # emulate code in infinite time and unlimited instructions
    mu.emu_start(ADDRESS, ADDRESS + len(ARM_BYTECODE))
    # now print out the R0 register
    print("Emulation done. Below is the result")
    # retrieve the result from the R1 register
    r1 = mu.reg_read(UC_ARM_REG_R1)
    print(">>  R1 = %u" % r1)
except UcError as e:
    print("Unicorn Error: %s" % e)