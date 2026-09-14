/* Hello, World for the neural machine: each character is emitted as a pixel record
   through the output port; the host decodes the port word's completions in order. */
int main(void) {
    out_pixel('H'); out_pixel('e'); out_pixel('l'); out_pixel('l'); out_pixel('o');
    out_pixel(','); out_pixel(' ');
    out_pixel('W'); out_pixel('o'); out_pixel('r'); out_pixel('l'); out_pixel('d'); out_pixel('!');
    return 0;
}
