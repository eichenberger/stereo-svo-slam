#ifndef OPENCVIMAGEPROVIDER_H
#define OPENCVIMAGEPROVIDER_H

#include <QAbstractVideoSurface>
#include <QImage>
#include <QPainter>
#include <QPixmap>

#include <opencv2/opencv.hpp>

#include "stereo_slam_types.hpp"

class OpenCVImageProvider : public QObject
{
    Q_OBJECT
    Q_PROPERTY( QAbstractVideoSurface* videoSurface READ videoSurface WRITE setVideoSurface )
public:
    OpenCVImageProvider(QObject *parent = nullptr, bool show_points = false);

    QAbstractVideoSurface* videoSurface() const;
    void setVideoSurface( QAbstractVideoSurface* s );

    void setImage(const Frame &frame);

signals:

public slots:

private:
    QPixmap pixmap;
    QAbstractVideoSurface* surface;
    bool show_points;
};

#endif // OPENCVIMAGEPROVIDER_H
