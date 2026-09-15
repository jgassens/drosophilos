/* A perspective column loop in the shape of Doom's r_segs (Stage E1 shape, 16-bit): for each
   of 8 columns the wall distance comes from the map indexed by the heading offset, the
   projected height is scale / distance done as a reciprocal table (Q8) and a multiply, then a
   shift back (no division on this machine). One pixel record (height) per column. */
static u16 map[8] = {6, 6, 5, 4, 3, 4, 5, 6};
static u16 recip[8] = {0, 256, 128, 85, 64, 51, 43, 37};   /* round(256 / d), d = 0..7 */
static u16 col, d, r, h, heading;

int main(void) {
    heading = in_read();
    col = 0;
    while (col != 8) {
        d = map[(col + heading) & 7];
        r = recip[d];
        h = (r * 100) >> 8;      /* scale 100: a full-height wall at distance 1 */
        out_pixel(h);
        col = col + 1;
    }
    return 0;
}
