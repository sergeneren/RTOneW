#pragma once

#include <QtWidgets/QMainWindow>
#include <QFileDialog>
#include <QTime>
#include <QImage>
#include <QPixmap>
#include <QDebug>
#include <QWheelEvent>
#include <QGraphicsScene>
#include <QGraphicsSceneEvent>
#include <QGraphicsPixmapItem>
#include "ui_mainWindow.h"
#include "customGV.h"

#include "float.h"


class vec3;

class mainWindow : public QMainWindow
{
	Q_OBJECT

public:
	mainWindow(QWidget *parent = Q_NULLPTR);
	void drawImage(QImage img, int currSpp, int goalSpp);
	customGraphicsView *myGraphicsView;
public slots:
	void onRenderButtonClicked();
	
	
signals:
	void RTOneW_process_signal(vec3 **pix, int s, int width, int height);	
	private slots:
	void RTOneW_process_slot(vec3 **pix, int s, int width, int height);
private:
	Ui::mainWindowClass ui;
	qreal h11 = 1.0, h22 = 0;
	QGraphicsPixmapItem pixmap;
	QImage image;
	bool eventFilter(QObject * obj, QEvent * event);
	void handleWheelOnGraphicsScene(QGraphicsSceneWheelEvent* event);

};
void send_to_render(int width, int height, int spp, float fov, float aperture);
void setInstanceForRenderDistribution(mainWindow* w);
void process_image(vec3 **pix, int s, int width, int height);