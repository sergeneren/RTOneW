#pragma once

#include <QGraphicsView>

#ifndef CUSTOMGV
#define CUSTOMGV



class customGraphicsView : public QGraphicsView {
public:
	customGraphicsView(QWidget* parent=0);

protected:
	void enterEvent(QEvent *event);
	void mousePressEvent(QMouseEvent *event);
	void mouseReleaseEvent(QMouseEvent *event);
	
};

#endif // !CUSTOMGV
