/* A Doom-like renderer with a moving player (Stage D + E1 shape): the tick kernel turns and
   moves the player from its input (bits 0-5 the turn, bits 8-9 forward / back), the column
   pass casts one ray per screen column through a 16 x 16 grid level in 4 steps of one cell
   (Q4.4 coordinates, trig tables), stores the projected wall height per column into a buffer,
   and the pixel pass paints ceiling, wall shaded by distance, or floor by comparing the pixel's
   row with the column's height read back from the buffer. Level, trig, height and shade
   tables are compiled in by the asset compiler. Screen 160 x 100, the horizon at row 50. */
static u16 grid[256] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
static u16 cost[64] = {16, 16, 16, 15, 15, 14, 13, 12, 11, 10, 9, 8, 6, 5, 3, 2, 0, 65534, 65533, 65531, 65530, 65528, 65527, 65526, 65525, 65524, 65523, 65522, 65521, 65521, 65520, 65520, 65520, 65520, 65520, 65521, 65521, 65522, 65523, 65524, 65525, 65526, 65527, 65528, 65530, 65531, 65533, 65534, 0, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 15, 16, 16};
static u16 sint[64] = {0, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 15, 16, 16, 16, 16, 16, 15, 15, 14, 13, 12, 11, 10, 9, 8, 6, 5, 3, 2, 0, 65534, 65533, 65531, 65530, 65528, 65527, 65526, 65525, 65524, 65523, 65522, 65521, 65521, 65520, 65520, 65520, 65520, 65520, 65521, 65521, 65522, 65523, 65524, 65525, 65526, 65527, 65528, 65530, 65531, 65533, 65534};
static u16 htab[5] = {4, 40, 20, 13, 10};
static u16 shade[5] = {7, 3, 4, 4, 5};
static u16 colmap[160] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19};
static u16 hbuf[160];
static u16 dbuf[160];
static u16 px, py, heading, in, turn, fwd, f, col, ang, dx, dy, x, y, cell, found, dist, h;
static u16 t, prow, pcol, top, bot, a, b, c, d, p;

int main(void) {
    px = 128; py = 128; heading = 0;   /* the middle of the level, looking along +x (Q4.4) */
    f = 2;
    while (f != 0) {
        col = 0;
        while (col != 160) {
            ang = (colmap[col] + heading) & 63;
            dx = cost[ang]; dy = sint[ang];
            x = px; y = py; found = 0; dist = 0;
        x = x + dx; y = y + dy;
        cell = grid[((y >> 4) << 4) | (x >> 4)];
        if (found == 0) { if (cell != 0) { found = 1; dist = 1; } }
        x = x + dx; y = y + dy;
        cell = grid[((y >> 4) << 4) | (x >> 4)];
        if (found == 0) { if (cell != 0) { found = 1; dist = 2; } }
        x = x + dx; y = y + dy;
        cell = grid[((y >> 4) << 4) | (x >> 4)];
        if (found == 0) { if (cell != 0) { found = 1; dist = 3; } }
        x = x + dx; y = y + dy;
        cell = grid[((y >> 4) << 4) | (x >> 4)];
        if (found == 0) { if (cell != 0) { found = 1; dist = 4; } }
            hbuf[col] = htab[dist];
            dbuf[col] = dist;
            col = col + 1;
        }
        p = 16000;
        while (p != 0) {
            t = in_read();
            pcol = t & 255;
            prow = t >> 8;
            h = hbuf[pcol];
            d = dbuf[pcol];
            top = 50 - h;
            bot = 50 + h;
            c = shade[d];
            a = (prow - top) & 32768;
            if (a != 0) { c = 1; }
            b = (bot - prow) & 32768;
            if (b != 0) { c = 2; }
            out_pixel(c);
            p = p - 1;
        }
        in = in_read();
        turn = in & 63;
        heading = (heading + turn) & 63;
        fwd = (in >> 8) & 3;
        if (fwd == 1) { px = px + cost[heading]; py = py + sint[heading]; }
        if (fwd == 2) { px = px - cost[heading]; py = py - sint[heading]; }
        f = f - 1;
    }
    return 0;
}
