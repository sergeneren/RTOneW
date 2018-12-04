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
	myGraphicsView->setGeometry(QRect(0, 0, 781, 781));
	myGraphicsView->setBaseSize(QSize(0, 0));
	myGraphicsView->viewport()->setProperty("cursor", QVariant(QCursor(Qt::CrossCursor)));
	myGraphicsView->setSceneRect(QRectF(0, 0, 0, 0));
	myGraphicsView->setAlignment(Qt::AlignCenter);
	myGraphicsView->setDragMode(QGraphicsView::ScrollHandDrag);
	myGraphicsView->setTransformationAnchor(QGraphicsView::AnchorUnderMouse);

	ui.detailView->setScene(new QGraphicsScene(this));
	ui.detailView->scene()->addItem(&detail_pixmap);


	connect(ui.renderButton, SIGNAL(clicked()), this, SLOT(onRenderButtonClicked()));
	connect(ui.actionExit, &QAction::triggered, qApp, QApplication::quit);
	connect(this, SIGNAL(RTOneW_process_signal(vec3*, int, int, int)), this, SLOT(RTOneW_process_slot(vec3*, int, int, int)));
	connect(myGraphicsView, &customGraphicsView::hoverSignal, this, &mainWindow::hoverSlot);
	ui.progressBar->setValue(0);
	pixmap.setAcceptHoverEvents(true);

	myGraphicsView->setScene(new QGraphicsScene(this));
	myGraphicsView->scene()->addItem(&pixmap);
	myGraphicsView->scene()->installEventFilter(this);

	//ui.renderButton->animateClick();

	ui.detail_frame->raise();
	ui.detail_frame->autoFillBackground();
	QGraphicsOpacityEffect *op = new QGraphicsOpacityEffect;
	op->setOpacity(0.7f);
	ui.detail_frame->setGraphicsEffect(op);
	ui.detail_frame->hide();
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

	return false;
}


void  mainWindow::drawImage(QImage img, int currSpp, int goalSpp) {

	image = img.copy();
	pixmap.setPixmap(QPixmap::fromImage(image));

	QApplication::processEvents();


	ui.progressBar->setValue(currSpp);

}

void mainWindow::RTOneW_process_slot(vec3 *pix, int s, int width, int height) {

	process_image(pix, s, width, height);

}

void mainWindow::hoverSlot(bool enter, QPointF *pos, QPoint *f) {

	if (enter) {
		if (pos->x() >= 0 && pos->x() < image.width() && pos->y() >= 0 && pos->y() < image.height()) {
			ui.detail_frame->show();
			QColor col = image.pixel(pos->x(), pos->y());

			ui.label_red->setText(QString::number(float(col.red()) / 255));
			ui.label_green->setText(QString::number(float(col.green()) / 255));
			ui.label_blue->setText(QString::number(float(col.blue()) / 255));

			ui.detailView->setTransform(QTransform(5, 0, 0, 0, 5, 0, 0, 0, 1));
			QImage detail_image(10, 10, QImage::Format_RGB32);
			int x_min = pos->x() - 5;
			int x_max = pos->x() + 5;
			int y_min = pos->y() - 5;
			int y_max = pos->y() + 5;

			//qDebug() << "x_min: " << x_min << "y_min: " << y_min;

			QColor black(0, 0, 0);
			
			for (int y = y_min; y < y_max; y++) {
				for (int x = x_min; x < x_max; x++) {

					QRgb rgb = black.rgb();
					if (x_min > -1 && y_min > -1 && y_max <= image.height() && x_max <= image.width()) rgb = image.pixel(x, y);
					
					detail_image.setPixel(x - x_min, y - y_min, rgb);
				}
			}

			detail_pixmap.setPixmap(QPixmap::fromImage(detail_image));


			QPoint l(f->x() + 25, f->y());
			ui.detail_frame->move(l);

		}
		else {

			ui.detail_frame->hide();

		}
	}
	else
	{
		ui.detail_frame->hide();
	}
}

void mainWindow::onRenderButtonClicked() {

	int width = 0;
	int height = 0;
	int spp = 0;
	int b_size = 0;
	int t_size = 0;
	float fov = 0;
	float aperture = 0;

	width = ui.width->text().toInt();
	height = ui.height->text().toInt();
	spp = ui.spp->text().toInt();
	fov = ui.fov->text().toFloat();
	aperture = ui.aperture->text().toFloat();
	b_size = ui.block_size->text().toInt();
	t_size = ui.thread_size->text().toInt();

	myGraphicsView->fitInView(0, 0, width, height);

	ui.progressBar->setRange(0, spp);

	if (ui.renderButton->text() == "Render") {
		ui.renderButton->setText("Cancel");
		send_to_render(width, height, spp, fov, aperture, b_size, t_size);
		ui.renderButton->setText("Render");
	}
	else {
		ui.renderButton->setText("Render");
		cancel_render();

	}


}