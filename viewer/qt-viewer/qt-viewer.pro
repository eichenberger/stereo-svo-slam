QT += qml quick

HEADERS +=
SOURCES += main.cpp

RESOURCES += qt-viewer.qrc

target.path = $$[QT_INSTALL_EXAMPLES]/qt-viewer
INSTALLS += target

OTHER_FILES += \
    main.qml

DISTFILES += \
    KeyFrames.qml \
    PointCloudViewer.qml \
    SideConfiguration.qml \
    Pose.qml
