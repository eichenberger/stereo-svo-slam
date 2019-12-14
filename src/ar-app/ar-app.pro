QT += 3dcore 3drender 3dinput 3dquick 3dlogic qml quick 3dquickextras widgets multimedia

SOURCES += \
    ../app/econ_input.cpp \
    ../app/image_input.cpp \
    main.cpp \
    opencvimageprovider.cpp \
    slam.cpp

OTHER_FILES += \
    main.qml

RESOURCES += \
    ar-app.qrc

HEADERS += \
    ../app/econ_input.hpp \
    ../app/image_input.hpp \
    opencvimageprovider.h \
    slam.h

DISTFILES += \
    AnimatedEntity.qml \
    ObjectEntity.qml

LIBS += -L/usr/local/lib -lopencv_core -lopencv_videoio -lopencv_imgproc -L../lib -lstereosvo -Wl,-rpath,../lib

INCLUDEPATH += /usr/local/include/opencv4 \
    ../include \
    ../app
