// A class defining and doing affine coordinate transformations
//
// If we ever want to support polar coordinates, I guess we should have 
// interface defining just transformPhysicalToLogical and 
// transformLogicalToPhysical.
//
// Copyright (c) 2000, 2004 Markus Demleitner
//  This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
//
// tabsize=2

import java.awt.Point;


class TwoDMatrix
{
	DoublePoint row1, row2;

	TwoDMatrix(DoublePoint row1, DoublePoint row2)
	{
		this.row1 = row1;
		this.row2 = row2;
	}

	public DoublePoint mul(DoublePoint vec)
	{
		return new DoublePoint(this.row1.scalTimes(vec),
			this.row2.scalTimes(vec));
	}

	public TwoDMatrix getInverse()
	{
		double a = this.row1.getX(), c = this.row1.getY();
		double b = this.row2.getX(), d = this.row2.getY();
		double normalizer = a*d-b*c;
		
		return new TwoDMatrix(new DoublePoint(d/normalizer, -b/normalizer),
			new DoublePoint(-c/normalizer, a/normalizer));
	}
}


/** 
 * AffineTrafos get defined by passing proper arguments to their constructor
 * and can then be queried for transformations between physical (screen) and
 * logical (defined by the gauges) coordinates.  We abbreviate logical
 * and physical coordinates lCoos and pCoos in the following.
 *
 * pCoos are always orthogonal and their axes intersect
 * in (0,0) as defined by the awt drawing model.
 *
 * lCoos can be a complete mess, i.e., the axes may be askew and their
 * zeroes need not be at the intersections points.
 *
 * pCoos are instances of the awt Point (because we don't know about fractional
 * pixels), lCoos are instances of DoublePoint.
 *
 * To transform from pCoos p to lCoos l, we do:
 *
 * l = lt + M*(p-pt)
 *
 * where M is the transformation matrix (i.e. the inverse of the unit
 * vectors of the lCoos expressed in pCoos), pt is the the intersection
 * point of the lines defined by the unit vectors (which would be their
 * zero point, obviously in pCoos), and lt undoes the translation of pt
 * in lCoos (it's just the intersection point in lCoos plus the starts of 
 * respective ranges).  Draw it up, it's, as I said, a bit messy.
 *
 * CooTrafos are immutable, i.e. if your transformation changes, just discard
 * your old CooTrafo and build a new one.  It doesn't take that much time but
 * helps avoiding nasty bugs.
 *
 * It probably would be nice to have separate classes of the various
 * combinations of logarithmic axes (the constructor already tells you
 * that this class rolls too much into one).  Then again, the resulting
 * inflation of classes would have a distinct smell of overengineering...
 */


public class AffineTrafo {

	private DoublePoint preTrafoTranslation=null;
	private DoublePoint postTrafoTranslation=null;
	private TwoDMatrix trafoMatrix=null;
	boolean isLogX, isLogY;

	/** 
	 * transforms a single Point in pCoos to lCoos
	 *
	 * @param xZeroPhys the location of the start of the x axis in pCoos
	 * @param yZeroPhys the location of the start of the y axis in pCoos
	 * @param xAxisPhys the location of the end of the x axis in pCoos
	 * @param yAxisPhys the location of the end of the y axis in pCoos
	 * @param xMin logical value of the start of the x axis
	 * @param xMax logical value of the end of the x axis
	 * @param yMin logical value of the start of the y axis
	 * @param yMax logical value of the end of the y axis
	 * @param isLogY true if y axis is logarithmic
	 */
	AffineTrafo(Point xZeroPhys, Point yZeroPhys,
			Point xAxisPhys, Point yAxisPhys,
			double xMin, double xMax,
			double yMin, double yMax,
			boolean isLogX, boolean isLogY) throws MissingData
	{ 
		this.isLogX = isLogX;
		this.isLogY = isLogY;
		if (isLogX) {
			xMin = Math.log(xMin);
			xMax = Math.log(xMax);
		}
		if (isLogY) {
			yMin = Math.log(yMin);
			yMax = Math.log(yMax);
		}
		this.checkArguments(xZeroPhys, yZeroPhys,
			xAxisPhys, yAxisPhys,
			xMin, xMax, yMin, yMax);

		this.computeTransform(new DoublePoint(xZeroPhys),
			new DoublePoint(yZeroPhys), new DoublePoint(xAxisPhys),
			new DoublePoint(yAxisPhys), xMin, xMax, yMin, yMax);
	}

	/**
	 * validates relevant arguments and raises a MissingData exception
	 * if it is not.
	 *
	 * The main problem we try to solve here is that data is not necessarily
	 * complete or sensible, since it may be rather unfiltered user input.  
	 */

	private void checkArguments(Point xZeroPhys, Point yZeroPhys,
			Point xAxisPhys, Point yAxisPhys,
			double xMin, double xMax,
			double yMin, double yMax) throws MissingData
	{
		if (xZeroPhys==null || yZeroPhys==null ||
				xAxisPhys==null || yAxisPhys==null) {
			throw new MissingData("You need to specify all axes\nto get results.");
		}
		// This happens in partiular with negative values on log axes.
		if (Double.isNaN(xMin) || Double.isNaN(xMax) ||
				Double.isNaN(yMin) || Double.isNaN(yMax)) {
			throw new MissingData("Invalid input for gauges.\n"+
				"With log axes, labels must be positive.");
		}
	}

	/**
	 * fills in the preTrafoTranslation, postTrafoTranslation and
	 * trafoMatrix attributes.
	 */
	private void computeTransform(DoublePoint x0, DoublePoint y0,
		DoublePoint x1, DoublePoint y1, double xMin, double xMax,
		double yMin, double yMax)
	{
		DoublePoint unitX = x1.
			minus(x0).
			times(1/(xMax-xMin));
		DoublePoint unitY = y1.
			minus(y0).
			times(1/(yMax-yMin));
		DoublePoint lIntersectionPoint = new TwoDMatrix(unitX.times(-1), unitY).
			getInverse().
			mul(x0.
				minus(y0));
		this.preTrafoTranslation = x0.
			plus(unitX.
				times(lIntersectionPoint.getX()));
		this.postTrafoTranslation = lIntersectionPoint.
			plus(new DoublePoint(xMin, yMin));
		this.trafoMatrix = new TwoDMatrix(unitX, unitY).getInverse();
	}

	/**
	 * returns logical coordinates for the screen coordinates physCoo
	 */
	public DoublePoint transformPhysicalToLogical(Point physCoo)
	{
		DoublePoint result = new DoublePoint(physCoo);

		result = this.trafoMatrix.
			mul(result.
				minus(this.preTrafoTranslation)
				).
			plus(this.postTrafoTranslation);
		if (this.isLogX) {
			result = new DoublePoint(Math.exp(result.getX()), result.getY());
		}
		if (this.isLogY) {
			result = new DoublePoint(result.getX(), Math.exp(result.getY()));
		}
		return result;
	}
}
