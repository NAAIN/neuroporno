nasm -f elf32 kernel.asm -o kasm.o
gcc -fno-stack-protector -m32 -c kernel.c -o bin/kc.o
ld -m elf_i386 -T link.ld -o kernel kasm.o bin/kc.o
qemu-system-i386 -kernel kernel

