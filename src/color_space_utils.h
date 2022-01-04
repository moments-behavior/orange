
/**
 * @brief Template class to facilitate color space conversion
 */
template <typename T>
void uyvy422packed_to_nv12(T *pFrame, T *pConvFrame, int nWidth, int nHeight, int nPitch = 0)
{
    if (nPitch == 0)
    {
        nPitch = nWidth;
    }

    // sizes of source surface plane
    int nSizePlaneY = nPitch * nHeight;
    int nSizePlaneU = ((nPitch + 1) / 2) * ((nHeight + 1) / 2);
    int nSizePlaneV = nSizePlaneU;

    // split chroma from interleave to planar
    for (int y = 0; y < nHeight; y++)
    {
        for (int x = 0; x < nWidth; x++)
        {
            py[y * nPitch + x] = pFrame[y * nPitch + x * 2 + 1];
            pQuad[y * ((nWidth + 1) / 2) + x] = puv[y * nPitch + x * 2 + 1];
        }
    }

    
}
