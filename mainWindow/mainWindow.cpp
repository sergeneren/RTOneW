#include "mainWindow.h"
#include <QFileDialog>
#include <iostream>
#include <algorithm>
#include <cstdint>

mainWindow::mainWindow(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
	
	myGraphicsView = new customGraphicsView(ui.scrollAreaWidgetContents);
	myGraphicsView->setObjectName(QStringLiteral("graphicsView"));
	myGraphicsView->setGeometry(QRect(5, 11, 771, 741));
	myGraphicsView->setBaseSize(QSize(0, 0));
	myGraphicsView->viewport()->setProperty("cursor", QVariant(QCursor(Qt::CrossCursor)));
	myGraphicsView->setSceneRect(QRectF(0, 0, 0, 0));
	myGraphicsView->setAlignment(Qt::AlignCenter);
	myGraphicsView->setDragMode(QGraphicsView::ScrollHandDrag);
	myGraphicsView->setTransformationAnchor(QGraphicsView::AnchorUnderMouse);

	connect(ui.renderButton, SIGNAL(clicked()), this, SLOT(onRenderButtonClicked()));
	connect(ui.actionExit, &QAction::triggered, qApp, QApplication::quit);
	connect(this, SIGNAL(RTOneW_process_signal(vec3*, int,int,int)), this, SLOT(RTOneW_process_slot(vec3*,int,int,int)));
	ui.progressBar->setValue(0);
	myGraphicsView->setScene(new QGraphicsScene(this));
	myGraphicsView->scene()->addItem(&pixmap);
	myGraphicsView->scene()->installEventFilter(this);
	
	//ui.renderButton->animateClick();
	
}


void mainWindow::handleWheelOnGraphicsScene(QGraphicsSceneWheelEvent* scrollevent)
{
	const int degrees = scrollevent->delta() / 8;
	int steps = degrees / 15;
	double scaleFactor = .5; //How fast we zoom
	const qreal minFactor = 1.0;
	const qreal maxFactor = 100.0;
	
	if (steps > 0)
	{
		h11 = (h11 >= maxFactor) ? h11 : (h11 + scaleFactor);
		h22 = (h22 >= maxFactor) ? h22 : (h22 + scaleFactor);
	}
	else
	{
		h11 = (h11 <= minFactor) ? minFactor : (h11 - scaleFactor);
		h22 = (h22 <= minFactor) ? minFactor : (h22 - scaleFactor);
	}
	myGraphicsView->setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
	myGraphicsView->setTransform(QTransform(h11, 0, 0, 0, h11, 0, 0, 0, 1));

}


bool mainWindow::eventFilter(QObject *obj, QEvent *event)
{
	if (event->type() == QEvent::GraphicsSceneWheel)
	{
		QGraphicsSceneWheelEvent *scrollevent = static_cast<QGraphicsSceneWheelEvent *>(event);
		handleWheelOnGraphicsScene(scrollevent);
		return true;
	}
	// Other events should propagate - what do you mean by propagate here?
	return false;
}


void  mainWindow::drawImage(QImage img, int currSpp, int goalSpp) {
	
	image = img.copy();
	pixmap.setPixmap(QPixmap::fromImage(image));
	
	/*
	myGraphicsView->setScene(new QGraphicsScene(this));
	myGraphicsView->scene()->setSceneRect(QRectF(QPointF(0, 0), QSizeF(512, 512)));
	myGraphicsView->scene()->addItem(&pixmap);
	myGraphicsView->scene()->installEventFilter(this);
	*/
	QApplication::processEvents();
	myGraphicsView->show();
	
	ui.progressBar->setValue(currSpp);

}

void mainWindow::RTOneW_process_slot(vec3 *pix, int s, int width, int height) {
	
	process_image(pix, s, width, height);
	
}


void mainWindow::onRenderButtonClicked() {
		
	int width = 0;
	int height = 0;
	int spp = 0; 

	float fov = 0;
	float aperture = 0;

	width = ui.width->text().toInt();
	height = ui.height->text().toInt();
	spp = ui.spp->text().toInt();
	fov = ui.fov->text().toFloat();
	aperture = ui.aperture->text().toFloat();


	ui.progressBar->setRange(0, spp);
	send_to_render(width, height, spp, fov, aperture);
}