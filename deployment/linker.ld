ENTRY(_start)

MEMORY
{
  FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 512K
  RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 128K
}

SECTIONS
{
  /* Code, constants, and vector table */
  .text :
  {
    KEEP(*(.isr_vector))
    *(.text*)
    *(.rodata*)
    *(.glue_7)
    *(.glue_7t)
  } > FLASH

  .ARM.extab : 
  {
    *(.ARM.extab*)
  } > FLASH

  .ARM.exidx : 
  {
    __exidx_start = .;
    *(.ARM.exidx*)
    __exidx_end = .;
  } > FLASH

  /* Initialized data, copied from FLASH to RAM */
  .data : AT (LOADADDR(.ARM.exidx) + SIZEOF(.ARM.exidx))
  {
    __data_start__ = .;
    *(.data*)
    __data_end__ = .;
  } > RAM

  /* Zero-initialized data */
  .bss :
  {
    __bss_start__ = .;
    *(.bss*)
    *(COMMON)
    __bss_end__ = .;
    _end = .;
    __end__ = .;
    end = .;
  } > RAM

  /* Stack */
  .stack (COPY):
  {
    _estack = .;
    . = . + 0x1000;
  } > RAM
}
