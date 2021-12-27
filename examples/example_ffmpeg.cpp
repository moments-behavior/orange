#include <stdio.h>
#include <iostream>
#include <iomanip>

using namespace std;


int main(int argc, char **argv) {
    FILE *pPipe;
    long lSize;
    char * imgdata;
    int imgcols = 640, imgrows = 480, elemSize = 3;

    imgdata = "/home/ash/Pictures/robot_home_position.jpg";

    stringstream sstm;
    sstm << "/usr/local/bin/ffmpeg -y -f rawvideo -vcodec rawvideo -s " << imgcols << "x" << imgrows  <<" -pix_fmt rgb24 -i - -shortest my_output.mp4";

    if ( !(pPipe = popen(sstm.str().c_str(), "w")) ) {
        cout << "popen error" << endl;
        exit(1);
    }

    // open a pipe to FFmpeg
    lSize = imgrows * imgcols * elemSize;

    // write to pipe
    fwrite(imgdata, 1, lSize, pPipe);
    fflush(pPipe);
    fclose(pPipe);

    return 0;

}

