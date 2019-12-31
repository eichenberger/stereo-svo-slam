QT += 3dcore 3drender 3dinput 3dquick 3dlogic qml quick 3dquickextras widgets multimedia

CONFIG += c++17
QMAKE_CXXFLAGS += -std=c++17 

SOURCES += \
    ../app/econ_input.cpp \
    ../app/video_input.cpp \
    ../app/image_input.cpp \
    ../app/slam_app.cpp \
    main.cpp \
    opencvimageprovider.cpp \
    slam.cpp

OTHER_FILES += \
    main.qml

RESOURCES += \
    ar-app.qrc

HEADERS += \
    ../app/econ_input.hpp \
    ../app/video_input.hpp \
    ../app/image_input.hpp \
    ../app/slam_app.hpp \
    opencvimageprovider.h \
    slam.h

DISTFILES += \
    AnimatedEntity.qml \
    ObjectEntity.qml

LIBS += -L$$(OPENCV_LIB_DIR) -lopencv_core -lopencv_videoio -lopencv_imgproc -L../lib -lstereosvo -Wl,-rpath,../lib

INCLUDEPATH += $$(OPENCV_INC_DIR) \
    ../include \
    ../app
