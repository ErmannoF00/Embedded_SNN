.syntax unified
.cpu cortex-m4
.fpu softvfp
.thumb

.global _start
.global Reset_Handler

.section .isr_vector, "a", %progbits
.type _start, %function
_start:
    .word  _estack            // Initial Stack Pointer
    .word  Reset_Handler      // Reset Handler

.section .text.Reset_Handler
.type Reset_Handler, %function
Reset_Handler:
    bl main
    b .

.size Reset_Handler, . - Reset_Handler

.section .stack
    .word 0x20001000
_estack:
