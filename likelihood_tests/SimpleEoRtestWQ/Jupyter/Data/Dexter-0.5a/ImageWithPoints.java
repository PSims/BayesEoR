// Mainly a class that has a background image, on top of which you can
// draw points and lines
//
// Copyright (c) 2000 Markus Demleitner
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

import java.awt.*;
import java.awt.event.*;
import java.util.Vector;
import java.util.Enumeration;
import java.util.Iterator;


public class ImageWithPoints extends Canvas
{ 
	Image figure;
	Vector<Datapoint> points;
	Gauge tmpgauge=null;
	Recogniser myRecogniser;
	public Gauge hgauge=null;
	public Gauge vgauge=null;
	Color pointColor=new Color(0,180,0);
	int imszx,imszy;
	boolean imageAvailable=false;
	ImageWithPoints me=this;  // this is "this" for the mouse handlers
	boolean userMouseActive=false;
	MagGlass magGlass=null;
	RecogniserSettings recSettings;
	DExtractor parent;
	Image backStore=null;
	static final long serialVersionUID=20060308L;

	ImageWithPoints(DExtractor parent, MagGlass magGlass,
		RecogniserSettings recSettings)
	{
		this.parent = parent;
		this.magGlass = magGlass;
		this.recSettings = recSettings;
		this.points = new Vector<Datapoint>();
		this.setCursor(new Cursor(Cursor.CROSSHAIR_CURSOR));
		this.addMouseListener(normalMouseAction);
		this.addMouseMotionListener(normalMouseMotion);
	}


	public void addNotify()
	{	
		super.addNotify();
		if (backStore==null)
			this.createBackStore();
	}


	public synchronized void setImage(Image im)
	{	
		figure = im;

		if (im.getWidth(this)==-1) { 
			this.backStore = null;
			this.magGlass.setImage(im);
			this.imageAvailable = false;
			Graphics g=getGraphics();

			if (g!=null) {	
				try {	
					g.setColor(Color.black);
					g.drawString("Retrieving image. Please stand by...", 10, 10);
				}
				finally {	
					g.dispose();
				}
			}
		}
		else {
			this.imageAvailable = true;
			createBackStore();
		}
		repaint();
	}


	public void setGauge(Gauge gauge)
	{	
		Rectangle uprect=null;

		if (gauge.horiz) {	
			if (this.hgauge!=null) {
				uprect = this.hgauge.bbox();
			}
			this.hgauge = gauge;
		}
		else {	
			if (this.vgauge!=null) {
				uprect = this.vgauge.bbox();
			}
			this.vgauge = gauge;
		}
		try {
			this.parent.computeTransform();
		} catch (Exception e) {
			// Failure is ok here (incomplete data are to be expected)
		}
		if (uprect!=null) {
			this.makeImage(uprect);
		}
		this.makeImage(gauge.bbox());
	}


	public void setGauge(Point p1, Point p2)
	{	
		Gauge mygauge=new Gauge(this, p1.x, p1.y, p2.x, p2.y);

		mygauge.normalise();
		this.setGauge(mygauge);
	}


	void paintPoints(Graphics g)
	{	
		for (Iterator<Datapoint> it=this.points.iterator(); it.hasNext();) {
			it.next().paint(g);
		}
	}


	void paintTmpGauge(Graphics g)
	{
		if (this.tmpgauge!=null)
			this.tmpgauge.paint(g);
	}


	void paintGauges(Graphics g)
	{	
		if (this.hgauge!=null)
			this.hgauge.paint(g);
		if (this.vgauge!=null)
			this.vgauge.paint(g);
	}

	
	public void makeImage(Rectangle bbox)
	{
		if (this.backStore==null)
			return;
		Graphics backGraphics=this.backStore.getGraphics();
		if (backGraphics==null)
			return;
		if (bbox!=null) {
			backGraphics.setClip(bbox.x, bbox.y, bbox.width, bbox.height);
		} else { 
			bbox = new Rectangle(0, 0, this.backStore.getWidth(this),
				this.backStore.getHeight(this));
			backGraphics.setClip(0, 0, bbox.width, bbox.height);
		}
		backGraphics.drawImage(figure,0,0,this);
		this.paintPoints(backGraphics);
		this.paintGauges(backGraphics);
		backGraphics.dispose();
		this.repaint(bbox.x, bbox.y, bbox.width, bbox.height);
	}


	public synchronized void paint(Graphics g)
	{	
		if (this.backStore!=null) {	
			g.drawImage(backStore, 0, 0, this);
			this.paintTmpGauge(g);
		}
		else { 
			g.drawImage(figure, 0, 0, this);
			this.paintPoints(g);
			this.paintGauges(g);
			this.paintTmpGauge(g);
			if (!this.imageAvailable) {	
				g.setColor(Color.black);
				g.drawString("Retrieving image. Please stand by...", 10, 10);
			}
		}
	}


	public void newScale(int oldscale, int newscale)
	{	
		if (this.vgauge!=null) {	
			this.vgauge.newScale(oldscale, newscale);
		}
		if (this.hgauge!=null) {	
			this.hgauge.newScale(oldscale, newscale);
		}
		for (Iterator<Datapoint> it=this.points.iterator(); it.hasNext();) {
			it.next().newScale(oldscale, newscale);
		}
	}


	public int findClosest(int x, int y)
	{	
		int ind=-1, i;
		double mindist, dist;
		Point pt;

		mindist = Double.MAX_VALUE;
		for (i=0; i<this.points.size(); i++) {
			pt = this.points.elementAt(i).getCenter();
			dist = ((double)x-pt.x)*((double)x-pt.x)+
				((double)y-pt.y)*((double)y-pt.y);
			if (dist<=mindist) { 
				ind = i;
				mindist = dist;
			}
		}
		return ind;
	}


	/* A helper for imageUpdate, containing painting ops needing
	synchronization */
	protected synchronized void plotNewLines(Image targIm,
		int x, int y, int width, int height)
	{
		Graphics g=getGraphics();
	
		if (g!=null) {	
			try {	
				g.setClip(x, y, width, height);
				g.drawImage(targIm, 0, 0, this);
				g.drawString("Retrieving image. Please stand by...", 10, 10);
			}
			finally {	
				g.dispose();
			}
		}
	}


	/* A helper for imageUpdate, containing state changing we want
	to protect with a mutex */
	protected synchronized void handleImageCompletion() {
		this.imageAvailable = true;
		this.createBackStore();
		this.repaint();
	}

	public boolean imageUpdate(Image targIm, int infoflags,
		int x, int y, int width, int height)
	{	

		if (targIm!=figure) {
			return false;
		}

		if ((infoflags&java.awt.image.ImageObserver.SOMEBITS)==
				java.awt.image.ImageObserver.SOMEBITS) {	
			this.plotNewLines(targIm, x, y, width, height);
		}

		if ((infoflags&java.awt.image.ImageObserver.ALLBITS)!=0) {
			this.handleImageCompletion();
			this.parent.resizeToPreferredSize();
			return false;
		}
		return true;
	}


	private void createBackStore()
	{	
		if (this.imageAvailable) { 
			backStore = createImage(figure.getWidth(this),
				figure.getHeight(this));
			if (backStore==null)
				return;
			this.magGlass.setImage(backStore);
			this.makeImage(null);
		}
	}


	public synchronized void update(Graphics g)
	{
		this.paint(g);
	}


	void beep() 
	{
  	Toolkit.getDefaultToolkit().beep();
	}
 

	public void userMouseToNormMouse()
	{	
		this.removeMouseListener(userInputMouse);
		this.removeMouseMotionListener(userInputMotion);
		this.userMouseActive = false;
		this.addMouseListener(normalMouseAction);
		this.addMouseMotionListener(normalMouseMotion);
		this.setCursor(new Cursor(Cursor.CROSSHAIR_CURSOR));
		this.parent.releaseStatusLine();
	}


	public void askUserPoint(String msg)
	{	
		this.parent.allocStatusLine(msg);
		this.removeMouseListener(normalMouseAction);
		this.removeMouseMotionListener(normalMouseMotion);
		this.addMouseListener(userInputMouse);
		this.addMouseMotionListener(userInputMotion);
		this.userMouseActive = true;
		this.setCursor(new Cursor(Cursor.HAND_CURSOR));
	}


	public void removePoint(Datapoint dp)
	{ Rectangle bbox=dp.bbox;

		this.points.removeElement(dp);	
		this.makeImage(bbox);
	}


 	public void startRecogniser(String recName) throws Exception
	{	
		if (this.myRecogniser!=null) {
			throw new Exception("There already is a recognizer running.");
		}
		if (recName=="LineTracer") {	
			LineTracer l=new LineTracer(this,recSettings);
			l.start();
			this.askUserPoint("Please click on the line you want to trace");
			this.myRecogniser = l;
		} else if (recName=="PointFinder") {	
			PointFinder l=new PointFinder(this,recSettings);
			l.start();
			this.askUserPoint("Please click on a template point");
			this.myRecogniser = l;
		} else if (recName=="AxisFinder") {	
			AutoAxisFinder l=new AutoAxisFinder(this,recSettings);
			l.start();
			l.putCoordinate(new Point(0,0));
			this.myRecogniser = l;
		} else {
			throw new Exception("Unknown recognizer: "+recName);
		}
	}


	public void recogniserStopped()
	{	
		if (this.userMouseActive) {
			this.userMouseToNormMouse();
		}
		this.myRecogniser = null;
		this.parent.recogniserStopped();
		this.repaint();
	}


	public void stopRecogniser()
	{
		if (this.myRecogniser==null)
			return;
		this.myRecogniser.stopRecogniser();
	}


	public void addPoint(Point p)
	{	Datapoint dp=new Datapoint(p.x,p.y,pointColor,this);

		this.points.addElement(dp);
		this.makeImage(dp.bbox);
	}


	public Dimension getMaximumSize()
	{	
		return getPreferredSize();
	}

	public Dimension getMinimumSize()
	{	
		return getPreferredSize();
	}

	public Dimension getPreferredSize()
	{
		if (this.imageAvailable)
			return new Dimension(this.figure.getWidth(this), 
				this.figure.getHeight(this));
		else
			return new Dimension(500, 300);
	}


	public Point makeParallelToAxes(Datapoint pt, Point targ)
	{	
		Gauge gauge=null;

		int dx=targ.x-pt.getScrX();
		int dy=targ.y-pt.getScrY();
		if (Math.abs(dx)>=Math.abs(dy))
			gauge = hgauge;
		else
			gauge = vgauge;
		if (gauge==null)
			return null;

		double xproj = (gauge.end.x-gauge.start.x)/gauge.length();
		double yproj = (gauge.end.y-gauge.start.y)/gauge.length();
		double projlen = xproj*dx+yproj*dy;
		return new Point((int)Math.round(pt.getScrX()+xproj*projlen),
			(int)Math.round(pt.getScrY()+yproj*projlen));
	}

	// returns an Array of references to the Datapoint objects -- consider them
	// read-only, because any changes you make won't be reflected in
	// the display (unless you know what you're doing)
	public Datapoint[] getPoints()
	{ 
		Datapoint ptArr[] = new Datapoint[this.points.size()];
		this.points.copyInto(ptArr);
		return ptArr;
	}

	// coordinate transformation is delegated to parent, because we don't
	// know about physical coordinates.  This is mainly a service for
	// Datapoints
	public DoublePoint transformPhysicalToLogical(Point p)
	{
		return this.parent.transformPhysicalToLogical(p);
	}
	
	public void delAllPoints()
	{
		this.points = new Vector<Datapoint>();
		this.makeImage(null);
	}

	
	public Image getRecImage()
	{
		return figure;
	}

	// TODO: Define a PositionListener interface and allow them to register.
	// Call registred PositionListeners here.
	protected void newMousePos(Point newPos)
	{	
		if (this.magGlass!=null) {
			this.magGlass.setCoords(newPos);
		}
		this.parent.displayMousePos(newPos);
	}


	public void print()
	{	
		for (Iterator<Datapoint> it=this.points.iterator(); it.hasNext();) {
			System.out.println(it.next().center);
		}
	}


	MouseAdapter normalMouseAction=new MouseAdapter()
	{
		public void mouseClicked(MouseEvent e) 
		{ 
			int x = e.getX();
			int y = e.getY();
			Enumeration<Datapoint> pts = points.elements();

			while (pts.hasMoreElements() && !e.isConsumed()) {
				((Datapoint)pts.nextElement()).mouseClicked(e);
			}
			if (e.isConsumed())
				return;
			if (!e.isShiftDown()&&
					((e.getModifiers()&InputEvent.BUTTON1_MASK)==
						InputEvent.BUTTON1_MASK)) {	
				addPoint(new Point(x,y));
				e.consume();
			}
		}

		public  void mousePressed(MouseEvent e) 
		{	
			Enumeration<Datapoint> pts = points.elements();

			if (hgauge!=null)
				hgauge.mousePressed(e);
			if (vgauge!=null && !e.isConsumed())
				vgauge.mousePressed(e);
			while (pts.hasMoreElements() && !e.isConsumed())
				((Datapoint)pts.nextElement()).mousePressed(e);
			if (e.isConsumed())
				return;
			tmpgauge = new Gauge(me,e.getX(),e.getY(),e.getX(),e.getY());
		}


		public  void mouseReleased(MouseEvent e) 
		{	
			Enumeration<Datapoint> pts = points.elements();

			if (hgauge!=null)
				hgauge.mouseReleased(e);
			if (vgauge!=null && !e.isConsumed())
				vgauge.mouseReleased(e);
			while (pts.hasMoreElements() && !e.isConsumed())
				((Datapoint)pts.nextElement()).mouseReleased(e);
			if (e.isConsumed())
				return;

			if (tmpgauge==null)
				return;
			if (tmpgauge.length()<20) { 
				Rectangle obox = tmpgauge.bbox();
				repaint(obox.x, obox.y, obox.width, obox.height);
				tmpgauge = null;
				return;
			}
			tmpgauge.setEnd(e.getX(), e.getY());
			setGauge(tmpgauge.start, tmpgauge.end);
			tmpgauge.paint(getGraphics());
			tmpgauge = null;
			e.consume();
		}
	};

	MouseMotionAdapter normalMouseMotion=new MouseMotionAdapter()
	{
		public void mouseDragged(MouseEvent e)
		{	
			Enumeration<Datapoint> pts = points.elements();

			newMousePos(e.getPoint());
			if (hgauge!=null)
				hgauge.mouseDragged(e);
			if (vgauge!=null && !e.isConsumed())
				vgauge.mouseDragged(e);
			while (pts.hasMoreElements() && !e.isConsumed())
				((Datapoint)pts.nextElement()).mouseDragged(e);
			if (e.isConsumed()||tmpgauge==null)
				return;

			Rectangle obox = tmpgauge.bbox();
			repaint(obox.x, obox.y, obox.width, obox.height);
			tmpgauge.setEnd(e.getX(),e.getY());
			tmpgauge.paint(getGraphics());
			e.consume();
		}

		public void mouseMoved(MouseEvent e) 
		{	
			newMousePos(e.getPoint());
		}
	};


	MouseAdapter userInputMouse=new MouseAdapter()
	{
		public void mouseClicked(MouseEvent e) {	
			if (myRecogniser!=null && myRecogniser.putCoordinate(e.getPoint()))
					return;
			userMouseToNormMouse();
		}

	};


	MouseMotionAdapter userInputMotion=new MouseMotionAdapter()
	{
		public void mouseMoved(MouseEvent e) {	
			newMousePos(e.getPoint());
		}
	};



}

// vi:ts=2:
