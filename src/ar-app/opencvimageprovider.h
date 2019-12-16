#ifndef OPENCVIMAGEPROVIDER_H
#define OPENCVIMAGEPROVIDER_H

#include <QAbstractVideoSurface>
#include <QImage>
#include <QPainter>
#include <QPixmap>

#include <opencv2/opencv.hpp>

class OpenCVImageProvider : public QObject
{
    Q_OBJECT
    Q_PROPERTY( QAbstractVideoSurface* videoSurface READ videoSurface WRITE setVideoSurface )
public:
    OpenCVImageProvider(QObject *parent = nullptr);

    QAbstractVideoSurface* videoSurface() const;
    void setVideoSurface( QAbstractVideoSurface* s );

    void setImage(cv::Mat image);

signals:

public slots:

private:
    QPixmap pixmap;
    QAbstractVideoSurface* surface;
};

#endif // OPENCVIMAGEPROVIDER_H
