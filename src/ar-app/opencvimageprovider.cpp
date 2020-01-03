#include "opencvimageprovider.h"

#include <vector>
#include <QImage>
#include <QDebug>
#include <QVideoSurfaceFormat>

using namespace cv;
using namespace std;

OpenCVImageProvider::OpenCVImageProvider(QObject *parent, bool show_points) : QObject(parent), surface(nullptr),
    show_points(show_points)
{
    pixmap = QPixmap(752, 480);
    pixmap.fill(QColor("white"));
}

QAbstractVideoSurface *OpenCVImageProvider::videoSurface() const
{
    return surface;
}

void OpenCVImageProvider::setVideoSurface(QAbstractVideoSurface *s)
{
    qDebug() << "set video surface";
    surface = s;

}

static void draw_keypoints(const Frame &frame, Mat &out)
{

    cvtColor(frame.stereo_image.left[0], out,  COLOR_GRAY2RGB);
    const vector<KeyPoint2d> &kps = frame.kps.kps2d;
    const vector<KeyPointInformation> &info = frame.kps.info;
    for (size_t i = 0; i < kps.size(); i++) {
        if (info[i].ignore_completely)
            continue;
        Point kp = Point(kps[i].x, kps[i].y);
        Scalar color (info[i].color.r, info[i].color.g, info[i].color.b);

        int marker = info[i].type == KP_FAST ? MARKER_CROSS : MARKER_SQUARE;

        cv::drawMarker(out, kp, color, marker);
    }
}


void OpenCVImageProvider::setImage(const Frame &frame)
{
    if (surface == nullptr)
        return;

    Mat image;

    if (show_points)
        draw_keypoints(frame, image);
    else
        cvtColor(frame.stereo_image.left[0], image,  COLOR_GRAY2RGB);
    QImage _image(image.data,
                  image.cols, image.rows,
                  static_cast<int>(image.step),
                  QImage::Format_RGB888);


    if (!surface->isActive()) {
//        QList<QVideoFrame::PixelFormat> formats = surface->supportedPixelFormats();
//        for (auto fmt:formats) {
//            qDebug() << "Supported: " << fmt;
//        }
        // There is only limted video pixel format support
        QVideoSurfaceFormat videoFormat(_image.size(), QVideoFrame::Format_RGB32);
        videoFormat = surface->nearestFormat(videoFormat);
        if (!surface->start(videoFormat))
            qDebug() << "Can't start surface: " << surface->error();
    }

    _image = _image.convertToFormat(QVideoFrame::imageFormatFromPixelFormat(surface->surfaceFormat().pixelFormat()));
    surface->present( QVideoFrame( _image ) );
}
