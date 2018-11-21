#include "customGV.h"

customGraphicsView::customGraphicsView(QWidget* parent) :QGraphicsView(parent){}

void customGraphicsView::enterEvent(QEvent *event) {

	QGraphicsView::enterEvent(event);
	viewport()->setCursor(Qt::CrossCursor);
	
}

void customGraphicsView::mousePressEvent(QMouseEvent *event) {

	QGraphicsView::mousePressEvent(event);
	viewport()->setCursor(Qt::ClosedHandCursor);

}

void customGraphicsView::mouseReleaseEvent(QMouseEvent *event) {

	QGraphicsView::mouseReleaseEvent(event);
	viewport()->setCursor(Qt::CrossCursor);

}