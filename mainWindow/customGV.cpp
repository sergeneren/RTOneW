#include "customGV.h"

customGraphicsView::customGraphicsView(QWidget* parent) :QGraphicsView(parent){
	setMouseTracking(true);
	viewport()->setMouseTracking(true);
}


void customGraphicsView::enterEvent(QEvent *event) {

	QGraphicsView::enterEvent(event);
	viewport()->setCursor(Qt::CrossCursor);
	
}

void customGraphicsView::leaveEvent(QEvent *event) {
	
	QGraphicsView::leaveEvent(event);
	QPointF pos(0, 0);
	QPoint f(0, 0);
	viewport()->setCursor(Qt::ArrowCursor);
	emit hoverSignal(false, &pos, &f);

}

void customGraphicsView::mousePressEvent(QMouseEvent *event) {

	QGraphicsView::mousePressEvent(event);
	viewport()->setCursor(Qt::ClosedHandCursor);

}

void customGraphicsView::mouseReleaseEvent(QMouseEvent *event) {

	QGraphicsView::mouseReleaseEvent(event);
	viewport()->setCursor(Qt::CrossCursor);

}

void customGraphicsView::mouseMoveEvent(QMouseEvent *event) {

	QGraphicsView::mouseMoveEvent(event);
	QPointF pos = mapToScene(event->pos());
	QPoint f = event->pos();
	emit hoverSignal(true, &pos, &f);

}