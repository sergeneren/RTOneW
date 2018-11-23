/********************************************************************************
** Form generated from reading UI file 'mainWindow.ui'
**
** Created by: Qt User Interface Compiler version 5.11.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGraphicsView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QProgressBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QScrollArea>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_mainWindowClass
{
public:
    QAction *actionSave;
    QAction *actionExit;
    QAction *actionZoom_in;
    QAction *actionZoom_out;
    QWidget *centralWidget;
    QScrollArea *scrollArea;
    QWidget *scrollAreaWidgetContents;
    QGraphicsView *graphicsView;
    QProgressBar *progressBar;
    QPushButton *renderButton;
    QLabel *label;
    QLabel *label_2;
    QLabel *label_3;
    QLabel *label_4;
    QLabel *label_5;
    QLabel *label_6;
    QFrame *line;
    QWidget *layoutWidget;
    QVBoxLayout *verticalLayout;
    QLineEdit *width;
    QLineEdit *height;
    QLineEdit *spp;
    QWidget *layoutWidget1;
    QVBoxLayout *verticalLayout_2;
    QLineEdit *fov;
    QLineEdit *aperture;
    QLabel *label_7;
    QLabel *label_8;
    QLineEdit *block_size;
    QLabel *label_9;
    QLineEdit *thread_size;
    QMenuBar *menuBar;
    QMenu *menuFile;
    QMenu *menuView;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *mainWindowClass)
    {
        if (mainWindowClass->objectName().isEmpty())
            mainWindowClass->setObjectName(QStringLiteral("mainWindowClass"));
        mainWindowClass->resize(1061, 914);
        actionSave = new QAction(mainWindowClass);
        actionSave->setObjectName(QStringLiteral("actionSave"));
        actionExit = new QAction(mainWindowClass);
        actionExit->setObjectName(QStringLiteral("actionExit"));
        actionZoom_in = new QAction(mainWindowClass);
        actionZoom_in->setObjectName(QStringLiteral("actionZoom_in"));
        actionZoom_out = new QAction(mainWindowClass);
        actionZoom_out->setObjectName(QStringLiteral("actionZoom_out"));
        centralWidget = new QWidget(mainWindowClass);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        scrollArea = new QScrollArea(centralWidget);
        scrollArea->setObjectName(QStringLiteral("scrollArea"));
        scrollArea->setGeometry(QRect(-1, -1, 781, 851));
        scrollArea->setWidgetResizable(true);
        scrollAreaWidgetContents = new QWidget();
        scrollAreaWidgetContents->setObjectName(QStringLiteral("scrollAreaWidgetContents"));
        scrollAreaWidgetContents->setGeometry(QRect(0, 0, 779, 849));
        graphicsView = new QGraphicsView(scrollAreaWidgetContents);
        graphicsView->setObjectName(QStringLiteral("graphicsView"));
        graphicsView->setGeometry(QRect(5, 11, 771, 741));
        graphicsView->setBaseSize(QSize(0, 0));
        graphicsView->viewport()->setProperty("cursor", QVariant(QCursor(Qt::CrossCursor)));
        graphicsView->setSceneRect(QRectF(0, 0, 0, 0));
        graphicsView->setAlignment(Qt::AlignCenter);
        graphicsView->setDragMode(QGraphicsView::ScrollHandDrag);
        graphicsView->setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
        scrollArea->setWidget(scrollAreaWidgetContents);
        progressBar = new QProgressBar(centralWidget);
        progressBar->setObjectName(QStringLiteral("progressBar"));
        progressBar->setGeometry(QRect(10, 850, 1061, 20));
        QFont font;
        font.setFamily(QStringLiteral("Full Sans LC 50"));
        progressBar->setFont(font);
        progressBar->setValue(24);
        renderButton = new QPushButton(centralWidget);
        renderButton->setObjectName(QStringLiteral("renderButton"));
        renderButton->setGeometry(QRect(820, 700, 181, 51));
        QFont font1;
        font1.setFamily(QStringLiteral("Full Sans LC 50"));
        font1.setPointSize(15);
        font1.setBold(true);
        font1.setItalic(false);
        font1.setUnderline(false);
        font1.setWeight(75);
        font1.setStrikeOut(false);
        renderButton->setFont(font1);
        renderButton->setMouseTracking(false);
        renderButton->setCheckable(false);
        renderButton->setAutoDefault(false);
        label = new QLabel(centralWidget);
        label->setObjectName(QStringLiteral("label"));
        label->setGeometry(QRect(821, 31, 81, 16));
        QFont font2;
        font2.setFamily(QStringLiteral("Full Sans LC 50"));
        font2.setPointSize(9);
        font2.setBold(true);
        font2.setWeight(75);
        label->setFont(font2);
        label_2 = new QLabel(centralWidget);
        label_2->setObjectName(QStringLiteral("label_2"));
        label_2->setGeometry(QRect(820, 60, 81, 16));
        label_2->setFont(font2);
        label_3 = new QLabel(centralWidget);
        label_3->setObjectName(QStringLiteral("label_3"));
        label_3->setGeometry(QRect(820, 90, 71, 16));
        label_3->setFont(font2);
        label_4 = new QLabel(centralWidget);
        label_4->setObjectName(QStringLiteral("label_4"));
        label_4->setGeometry(QRect(850, 140, 121, 16));
        QFont font3;
        font3.setFamily(QStringLiteral("Full Sans LC 50"));
        font3.setPointSize(12);
        font3.setBold(true);
        font3.setItalic(false);
        font3.setUnderline(false);
        font3.setWeight(75);
        label_4->setFont(font3);
        label_5 = new QLabel(centralWidget);
        label_5->setObjectName(QStringLiteral("label_5"));
        label_5->setGeometry(QRect(820, 170, 91, 16));
        label_5->setFont(font2);
        label_6 = new QLabel(centralWidget);
        label_6->setObjectName(QStringLiteral("label_6"));
        label_6->setGeometry(QRect(820, 200, 81, 16));
        label_6->setFont(font2);
        line = new QFrame(centralWidget);
        line->setObjectName(QStringLiteral("line"));
        line->setGeometry(QRect(790, 113, 271, 20));
        line->setFrameShape(QFrame::HLine);
        line->setFrameShadow(QFrame::Sunken);
        layoutWidget = new QWidget(centralWidget);
        layoutWidget->setObjectName(QStringLiteral("layoutWidget"));
        layoutWidget->setGeometry(QRect(916, 31, 135, 74));
        verticalLayout = new QVBoxLayout(layoutWidget);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        width = new QLineEdit(layoutWidget);
        width->setObjectName(QStringLiteral("width"));

        verticalLayout->addWidget(width);

        height = new QLineEdit(layoutWidget);
        height->setObjectName(QStringLiteral("height"));

        verticalLayout->addWidget(height);

        spp = new QLineEdit(layoutWidget);
        spp->setObjectName(QStringLiteral("spp"));

        verticalLayout->addWidget(spp);

        layoutWidget1 = new QWidget(centralWidget);
        layoutWidget1->setObjectName(QStringLiteral("layoutWidget1"));
        layoutWidget1->setGeometry(QRect(920, 170, 135, 48));
        verticalLayout_2 = new QVBoxLayout(layoutWidget1);
        verticalLayout_2->setSpacing(6);
        verticalLayout_2->setContentsMargins(11, 11, 11, 11);
        verticalLayout_2->setObjectName(QStringLiteral("verticalLayout_2"));
        verticalLayout_2->setContentsMargins(0, 0, 0, 0);
        fov = new QLineEdit(layoutWidget1);
        fov->setObjectName(QStringLiteral("fov"));

        verticalLayout_2->addWidget(fov);

        aperture = new QLineEdit(layoutWidget1);
        aperture->setObjectName(QStringLiteral("aperture"));

        verticalLayout_2->addWidget(aperture);

        label_7 = new QLabel(centralWidget);
        label_7->setObjectName(QStringLiteral("label_7"));
        label_7->setGeometry(QRect(820, 314, 71, 20));
        label_7->setFont(font2);
        label_8 = new QLabel(centralWidget);
        label_8->setObjectName(QStringLiteral("label_8"));
        label_8->setGeometry(QRect(820, 344, 91, 20));
        label_8->setFont(font2);
        block_size = new QLineEdit(centralWidget);
        block_size->setObjectName(QStringLiteral("block_size"));
        block_size->setGeometry(QRect(920, 314, 133, 20));
        label_9 = new QLabel(centralWidget);
        label_9->setObjectName(QStringLiteral("label_9"));
        label_9->setGeometry(QRect(860, 270, 121, 16));
        label_9->setFont(font3);
        thread_size = new QLineEdit(centralWidget);
        thread_size->setObjectName(QStringLiteral("thread_size"));
        thread_size->setGeometry(QRect(920, 340, 133, 20));
        mainWindowClass->setCentralWidget(centralWidget);
        layoutWidget->raise();
        layoutWidget->raise();
        scrollArea->raise();
        progressBar->raise();
        renderButton->raise();
        label->raise();
        label_2->raise();
        label_3->raise();
        label_4->raise();
        label_5->raise();
        label_6->raise();
        line->raise();
        label_7->raise();
        label_8->raise();
        block_size->raise();
        label_9->raise();
        thread_size->raise();
        menuBar = new QMenuBar(mainWindowClass);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 1061, 21));
        QFont font4;
        font4.setFamily(QStringLiteral("Futura Md BT"));
        menuBar->setFont(font4);
        menuFile = new QMenu(menuBar);
        menuFile->setObjectName(QStringLiteral("menuFile"));
        menuFile->setFont(font4);
        menuView = new QMenu(menuBar);
        menuView->setObjectName(QStringLiteral("menuView"));
        menuView->setFont(font4);
        mainWindowClass->setMenuBar(menuBar);
        statusBar = new QStatusBar(mainWindowClass);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        mainWindowClass->setStatusBar(statusBar);

        menuBar->addAction(menuFile->menuAction());
        menuBar->addAction(menuView->menuAction());
        menuFile->addAction(actionSave);
        menuFile->addAction(actionExit);
        menuView->addAction(actionZoom_in);
        menuView->addAction(actionZoom_out);

        retranslateUi(mainWindowClass);

        QMetaObject::connectSlotsByName(mainWindowClass);
    } // setupUi

    void retranslateUi(QMainWindow *mainWindowClass)
    {
        mainWindowClass->setWindowTitle(QApplication::translate("mainWindowClass", "RTOneW", nullptr));
        actionSave->setText(QApplication::translate("mainWindowClass", "Save", nullptr));
        actionExit->setText(QApplication::translate("mainWindowClass", "Exit", nullptr));
        actionZoom_in->setText(QApplication::translate("mainWindowClass", "Zoom in", nullptr));
        actionZoom_out->setText(QApplication::translate("mainWindowClass", "Zoom out", nullptr));
        renderButton->setText(QApplication::translate("mainWindowClass", "Render", nullptr));
        label->setText(QApplication::translate("mainWindowClass", "Width", nullptr));
        label_2->setText(QApplication::translate("mainWindowClass", "Height", nullptr));
        label_3->setText(QApplication::translate("mainWindowClass", "Samples", nullptr));
        label_4->setText(QApplication::translate("mainWindowClass", "Camera Options", nullptr));
        label_5->setText(QApplication::translate("mainWindowClass", "FOV", nullptr));
        label_6->setText(QApplication::translate("mainWindowClass", "Aperture", nullptr));
        width->setText(QApplication::translate("mainWindowClass", "512", nullptr));
        height->setText(QApplication::translate("mainWindowClass", "512", nullptr));
        spp->setText(QApplication::translate("mainWindowClass", "100", nullptr));
        fov->setText(QApplication::translate("mainWindowClass", "35", nullptr));
        aperture->setText(QApplication::translate("mainWindowClass", "0.25", nullptr));
        label_7->setText(QApplication::translate("mainWindowClass", "Block Size", nullptr));
        label_8->setText(QApplication::translate("mainWindowClass", "Thread Size", nullptr));
        block_size->setText(QApplication::translate("mainWindowClass", "8", nullptr));
        label_9->setText(QApplication::translate("mainWindowClass", "Cuda Options", nullptr));
        thread_size->setText(QApplication::translate("mainWindowClass", "8", nullptr));
        menuFile->setTitle(QApplication::translate("mainWindowClass", "File", nullptr));
        menuView->setTitle(QApplication::translate("mainWindowClass", "View", nullptr));
    } // retranslateUi

};

namespace Ui {
    class mainWindowClass: public Ui_mainWindowClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
