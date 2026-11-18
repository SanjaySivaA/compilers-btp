#include <stdint.h>

__attribute__((section(".dtcm_output"))) volatile uint32_t result_buffer[8];

int main(void) {
    uint32_t accum = 0;
    for (int i = 1; i <= 10; ++i) {
        accum += i * 5;
    }

    result_buffer[0] = 0xDEADBEEF; // Status magic word
    result_buffer[1] = accum;      // Computation result (275 / 0x0113)
    result_buffer[2] = 42;

    return 0; // Control returns to _start, executing ebreak
}