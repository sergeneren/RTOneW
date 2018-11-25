#pragma once

#include <QObject>
#include <QGraphicsView>
#include <QGraphicsPixmapItem>
#include <QMouseEvent>
#include <QRgb>

#ifndef CUSTOMGV
#define CUSTOMGV



class customGraphicsView : public QGraphicsView{
	Q_OBJECT
public:
	customGraphicsView(QWidget* parent=0);
Q_SIGNALS:
	void hoverSignal(bool enter, QPointF *pos , QPoint *f);
	
protected:
	virtual void leaveEvent(QEvent *event);
	virtual void enterEvent(QEvent *event);
	virtual void mousePressEvent(QMouseEvent *event);
	virtual void mouseReleaseEvent(QMouseEvent *event);
	virtual void mouseMoveEvent(QMouseEvent *event);
	
};


#endif // !CUSTOMGV
