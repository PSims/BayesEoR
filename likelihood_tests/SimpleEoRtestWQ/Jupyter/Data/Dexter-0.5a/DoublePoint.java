import java.awt.Point;

/**
 * DoublePoints are used as both vectors and data points here, which is
 * why they have all those funny vector methods.
 *
 * They should be considered immutable.  All algebraic operations always
 * return new DoublePoints.
 */

class DoublePoint 
{
	protected double x, y;

	DoublePoint(double x, double y)
	{
		this.x = x;
		this.y = y;
	}

	DoublePoint(DoublePoint p)
	{
		this.x = p.x;
		this.y = p.y;
	}

	DoublePoint(Point p)
	{
		this.x = p.x;
		this.y = p.y;
	}

	double getX()
	{
		return this.x;
	}

	double getY()
	{
		return this.y;
	}

	DoublePoint plus(DoublePoint p)
	{
		return new DoublePoint(this.x+p.x, this.y+p.y);
	}

	DoublePoint minus(DoublePoint p)
	{
		return new DoublePoint(this.x-p.x, this.y-p.y);
	}

	DoublePoint times(double lambda)
	{
		return new DoublePoint(this.x*lambda, this.y*lambda);
	}

	double scalTimes(DoublePoint other)
	{
		return this.x*other.x+this.y*other.y;
	}
}
