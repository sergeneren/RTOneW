#include "mainWindow.h"
#include <QtWidgets/QApplication>
#include <QStyleFactory>

void setStyleSheet() {
	qApp->setStyle(QStyleFactory::create("Fusion"));

	QPalette darkPalette;
	darkPalette.setColor(QPalette::Window, QColor(45, 45, 45));
	darkPalette.setColor(QPalette::WindowText, QColor(200, 200, 200));
	darkPalette.setColor(QPalette::Base, QColor(25, 25, 27));
	darkPalette.setColor(QPalette::AlternateBase, QColor(53, 53, 53));
	darkPalette.setColor(QPalette::ToolTipBase, Qt::white);
	darkPalette.setColor(QPalette::ToolTipText, Qt::white);
	darkPalette.setColor(QPalette::Text, QColor(200, 200, 200));
	darkPalette.setColor(QPalette::Button, QColor(20, 115, 130));
	darkPalette.setColor(QPalette::ButtonText, QColor(200, 200, 200));
	darkPalette.setColor(QPalette::BrightText, QColor(100, 100, 100));
	darkPalette.setColor(QPalette::Link, QColor(20, 115, 130));
	darkPalette.setColor(QPalette::Shadow, QColor(11, 19, 22));
	darkPalette.setColor(QPalette::Midlight, QColor(45, 45, 45));
	darkPalette.setColor(QPalette::Dark, QColor(11, 11, 11));
	darkPalette.setColor(QPalette::Light, QColor(23, 37, 51));

	darkPalette.setColor(QPalette::Highlight, QColor(85, 107, 121));
	darkPalette.setColor(QPalette::HighlightedText, Qt::black);

	qApp->setPalette(darkPalette);
}

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	mainWindow w;
	setInstanceForRenderDistribution(&w);
	setStyleSheet();

	w.show();
	return a.exec();
}
