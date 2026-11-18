ENTRY(_start)

MEMORY
{
  ITCM (rx)  : ORIGIN = 0x00000000, LENGTH = 8K
  DTCM (rw)  : ORIGIN = 0x00010000, LENGTH = 32K
}

SECTIONS
{
    .text : {
        *(.text._start)
        *(.text*)
    } > ITCM

    .dtcm_output : {
        *(.dtcm_output)
    } > DTCM

    .data : {
        *(.data*)
    } > DTCM

    .bss : {
        *(.bss*)
    } > DTCM
}