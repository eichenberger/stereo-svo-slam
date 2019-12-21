/****************************************************************************
**
** Copyright (C) 2014 Klaralvdalens Datakonsult AB (KDAB).
** Contact: https://www.qt.io/licensing/
**
** This file is part of the Qt3D module of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:BSD$
** Commercial License Usage
** Licensees holding valid commercial Qt licenses may use this file in
** accordance with the commercial license agreement provided with the
** Software or, alternatively, in accordance with the terms contained in
** a written agreement between you and The Qt Company. For licensing terms
** and conditions see https://www.qt.io/terms-conditions. For further
** information use the contact form at https://www.qt.io/contact-us.
**
** BSD License Usage
** Alternatively, you may use this file under the terms of the BSD license
** as follows:
**
** "Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are
** met:
**   * Redistributions of source code must retain the above copyright
**     notice, this list of conditions and the following disclaimer.
**   * Redistributions in binary form must reproduce the above copyright
**     notice, this list of conditions and the following disclaimer in
**     the documentation and/or other materials provided with the
**     distribution.
**   * Neither the name of The Qt Company Ltd nor the names of its
**     contributors may be used to endorse or promote products derived
**     from this software without specific prior written permission.
**
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
** OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
** LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
** DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
** THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
** (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
**
** $QT_END_LICENSE$
**
****************************************************************************/

#include <QGuiApplication>
#include <QQuickView>

#include "opencvimageprovider.h"

#include <iostream>
#include <fstream>
#include <vector>

#include <opencv2/opencv.hpp>

#include <QtWidgets/QApplication>
#include <QtCore/QTimer>
#include <QtCore/QCommandLineParser>
#include <QtCore/QCommandLineOption>
#include <QtCore/QFile>
#include <QtCore/QTextStream>
#include <QQuickItem>
#include <QQmlEngine>
#include <QQmlContext>
#include <QVariant>

#include "econ_input.hpp"
#include "slam.h"

using namespace cv;
using namespace std;

static OpenCVImageProvider *opencvImageProvider;

static void read_image(ImageInput *input, Slam *slam)
{
    Mat image;
    float time_stamp;

    Mat gray_r, gray_l;
    if (!input->read(gray_l, gray_r, time_stamp)) {
        QApplication::quit();
        return;
    }
    slam->new_image(gray_l, gray_r, time_stamp);
    opencvImageProvider->setImage(gray_l);
}

int main(int argc, char **argv)
{
    QApplication app(argc, argv);

    QCommandLineParser parser;
    QStringList arguments = app.arguments();
    parser.setApplicationDescription("SVO stereo SLAM application");
    parser.addHelpOption();
    parser.addPositionalArgument("camera", "The camera type to use can be econ");
    parser.addOptions({
            {{"v", "video"}, "Path to camera or video (/dev/videoX, video.mov)", "video"},
            {{"s", "settings"}, "Path to the settings file (Econ.yaml)", "settings"},
            {{"r", "hidraw"}, "econ: HID device to control the camera (/dev/hidrawX)", "hidraw"},
            {{"i", "hidrawimu"}, "econ: HID device to read the imu values(/dev/hidrawX)", "hidrawimu"},
            {{"e", "exposure"}, "econ: The exposure for the camera 1-30000", "exposure"},
            {{"d", "hdr"}, "econ: Use HDR video"},
            });


    parser.process(arguments);
    arguments.pop_back();

    QString camera_type = parser.positionalArguments().at(0);
    ImageInput *input;
    if (camera_type == "econ") {
        if (!parser.isSet("video") ||
                !parser.isSet("hidraw") ||
                !parser.isSet("settings") ||
                !parser.isSet("exposure")) {
            cout << "Please set all inputs for econ" << endl;
            cout << parser.helpText().toStdString() << endl;
            return -1;
        }

        EconInput *econ = new EconInput(parser.value("video").toStdString(),
                parser.value("hidraw").toStdString(),
                parser.value("settings").toStdString(),
                parser.value("hidrawimu").toStdString());
        econ->set_hdr(parser.isSet("hdr"));
        econ->set_manual_exposure(parser.value("exposure").toInt());

        input = econ;
    }
    else {
        cout << "Unknown camera type " << camera_type.toStdString() << endl;
        return -2;
    }

    CameraSettings camera_settings;
    input->get_camera_settings(camera_settings);

    Slam slam(camera_settings);

    QCommandLineOption showProgressOption("p", QCoreApplication::translate("main", "Show progress during copy"));
    parser.addOption(showProgressOption);

    QTimer timer;
    timer.setInterval(1.0/60.0*1000.0);

    QQuickView view;
    view.setResizeMode(QQuickView::SizeRootObjectToView);

    opencvImageProvider = new OpenCVImageProvider;

    qmlRegisterType<OpenCVImageProvider>("OpenCVImageProvider", 1, 0, "OpenCVImageProvider");

    QQmlEngine *engine = view.engine();

    QQmlContext *root = engine->rootContext();
    root->setContextProperty("mediaplayer", opencvImageProvider);
    root->setContextProperty("slam", &slam);

    view.setSource(QUrl("qrc:/main.qml"));
    view.show();

    QObject::connect(&timer, &QTimer::timeout,
            std::bind(&read_image, input, &slam));
    timer.start();

    app.exec();

    delete opencvImageProvider;
    delete input;
}



