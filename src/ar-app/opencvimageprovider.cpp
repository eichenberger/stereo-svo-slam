#include "opencvimageprovider.h"

#include <QImage>
#include <QDebug>
#include <QVideoSurfaceFormat>

OpenCVImageProvider::OpenCVImageProvider(QObject *parent) : QObject(parent), surface(nullptr)
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

void OpenCVImageProvider::setImage(cv::Mat image)
{
    if (surface == nullptr)
        return;

    QImage _image(image.data,
                  image.cols, image.rows,
                  static_cast<int>(image.step),
                  QImage::Format_Grayscale8 );


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
