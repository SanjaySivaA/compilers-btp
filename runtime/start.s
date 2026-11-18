.section .text._start
.global _start

_start:
    li sp, 0x18000
    call main

    /* Explicit halt loop after main returns */
    j halt

.global trap_handler
trap_handler:
    /* If ebreak triggers a trap, catch it here and halt */
    j halt

halt:
    ebreak
    j halt