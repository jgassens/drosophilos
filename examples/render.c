/* A toy renderer in the shape of Doom's r_segs column loop (Stage E1 shape): for each of
   8 screen columns, the wall distance comes from a map array indexed by the player's
   heading offset, the wall height from a lookup table (no division on this machine), and
   the column's pixel record (height) is emitted; a 9th record is the frame number. */
static u8 map[8] = {6, 6, 5, 4, 3, 4, 5, 6};
static u8 htab[8] = {100, 90, 75, 60, 45, 30, 20, 10};
static u8 col, d, h, heading, frame;

int main(void) {
    heading = in_read();
    frame = 0;
    while (frame != 2) {
        col = 0;
        while (col != 8) {
            d = map[(col + heading) & 7];
            h = htab[d];
            out_pixel(h);
            col = col + 1;
        }
        out_pixel(frame);
        heading = heading + 1;
        frame = frame + 1;
    }
    return 0;
}
