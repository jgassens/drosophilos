/* A Doom-like frame rendered one pixel per token (Stage E1 shape, 16-bit): the token packs
   the pixel's column and row; the wall distance for the column's view angle comes from a
   64-entry table compiled from the level's walls (a room with a doorway and a pillar), the
   projected wall height from a reciprocal table, and the pixel's colour is ceiling, wall
   (shaded by distance) or floor by comparing the row with the wall's top and bottom. The
   heading is the view angle in 64ths of a turn. The screen is 160 x 100 (the horizon at row
   50); a frame is 16,000 tokens dealt to as many copies of the kernel as there are brains, or
   a subsample of them for a small picture. */
static u16 wallmap[64] = {3, 8, 8, 8, 9, 9, 10, 9, 8, 8, 7, 7, 6, 7, 10, 12, 12, 12, 10, 7, 6, 7, 7, 8, 8, 9, 10, 9, 9, 8, 8, 8, 8, 8, 8, 8, 9, 9, 10, 9, 8, 8, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 8, 8, 9, 10, 3, 3, 3, 3, 3};
static u16 htab[16] = {40, 40, 20, 13, 10, 8, 7, 6, 5, 4, 4, 4, 3, 3, 3, 3};
static u16 shade[16] = {3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7};
static u16 t, col, row, heading, ang, d, h, top, bot, a, b, c, i;

int main(void) {
    heading = in_read();
    i = 4;
    while (i != 0) {
        t = in_read();
        col = t & 255;
        row = t >> 8;
        ang = ((col >> 3) + heading) & 63;   /* 20 angle steps across 160 columns: a wide view */
        d = wallmap[ang];
        h = htab[d];
        top = 50 - h;
        bot = 50 + h;
        c = shade[d];
        a = (row - top) & 32768;     /* row above the wall's top: the sign bit of row - top */
        if (a != 0) { c = 1; }       /* ceiling */
        b = (bot - row) & 32768;     /* row below the wall's bottom */
        if (b != 0) { c = 2; }       /* floor */
        out_pixel(c);
        i = i - 1;
    }
    return 0;
}
