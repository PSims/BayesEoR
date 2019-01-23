// The class that handles the main data extraction window, plus a
// few helper classes
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

import java.awt.*;
import java.awt.event.*;
import java.util.*;


// The text fields (and the log box) to specify beginning and and
// of one gauge -- may be either horizontal or vertical.
class GaugeGauge extends Panel
	implements TextListener
{	
	TextField start, end;
	int naturalx=0, naturaly=0;
	boolean vertical;
	GridBagLayout layout = new GridBagLayout();
	Checkbox islog;
	DExtractor parent;
	static final long serialVersionUID=20060308L;

	private class NumOnly extends KeyAdapter
	{	
		public void keyPressed(KeyEvent e)
		{	
			char c=e.getKeyChar();
		
			if (c>=' ')
				if (!((c>='0'&&c<='9')||c=='-'||c=='+'||c=='e'||c=='.'||c=='E'))
					e.consume();
		}
	}

	GaugeGauge(String variable, Color bgcolor, boolean vertical,
		DExtractor parent)
	{	
		GridBagConstraints c = new GridBagConstraints();
		Label lab;
		String labels[] = {"0", "1"};
		int labelalign;
		TextField nf;

		this.vertical = vertical;
		this.parent = parent;
		setForeground(bgcolor);
		setLayout(layout);
		start = new TextField("", 8);
		end = new TextField("", 8);
		islog = new Checkbox("log", false);

		if (vertical) {	
			String tmp;
			tmp = labels[0];
			labels[0] = labels[1];
			labels[1] = tmp;
			c.fill = GridBagConstraints.BOTH;
			c.gridwidth = GridBagConstraints.REMAINDER;
			labelalign = Label.LEFT;
		}
		else {	
			c.fill = GridBagConstraints.HORIZONTAL;
			labelalign = Label.RIGHT;
		}	
		c.weighty = 0.5;
		c.weightx = 0;
		lab = new Label(variable+labels[0], labelalign);
		layout.setConstraints(lab, c);
		add(lab);

		c.weightx = 0.5;
		c.weighty = 0.5;
		if (vertical)
			nf = end;
		else
			nf = start;
		layout.setConstraints(nf, c);
		add(nf);
		nf.addKeyListener(new NumOnly());
		nf.addTextListener(this);

		int oldfill = c.fill;
		c.fill = GridBagConstraints.NONE;
		layout.setConstraints(islog, c);
		add(islog);
		c.fill = oldfill;

		lab = new Label(variable+labels[1], labelalign);
		c.weightx = 0;
		c.weighty = 0.5;
		layout.setConstraints(lab, c);
		add(lab);

		c.weightx = 0.5;
		c.weighty = 0.5;
		if (vertical)
			nf = start;
		else
			nf = end;
		layout.setConstraints(nf, c);
		add(nf);
		nf.addKeyListener(new NumOnly());
		nf.addTextListener(this);
	}

	public double getstart()
	{
		return new Double(this.start.getText()).doubleValue();
	}

	public double getend()
	{
		return new Double(this.end.getText()).doubleValue();
	}

	public int computeUsefulSignificantDigits()
	{
		try {
			return Math.max(4, 
				(int)Math.round(-Math.log(
					Math.abs(this.getend()-this.getstart())/this.getstart())/2.3
				+4));
		} catch (NumberFormatException e) {
			return 4;
		}
	}

	public boolean isLogAxis()
	{	
		return islog.getState();
	}

	public Dimension getPreferredSize()
	{
		return layout.preferredLayoutSize((Container)this);
	}

	public Dimension getMinimumSize()
	{	
		return getPreferredSize();
	}

	public void textValueChanged(TextEvent ev)
	{	
		try {
			this.parent.computeTransform();
		} catch (CantComputeException e) {
		}
	}
}


// This handles the window in which most of the work gets done.
// Clearly, this should be split into at least three subclasses.
public class DExtractor extends Frame
	implements ActionListener, ComponentListener
{ 

	class Steeringstuff extends Panel
	{	
		int numbut;
		Button buttons[];
		static final long serialVersionUID=20060308L;


		// a container for some buttons, mainly here because I believe I am
		// too dumb for the awt layout managers
		Steeringstuff(Button buttons_p[], int numbut_p, ActionListener l)
		{	
			int i;

			numbut = numbut_p;
			buttons = buttons_p;
			setLayout(new GridLayout(numbut, 1));
			for (i=0; i<numbut; i++) {	
				add(buttons[i]);
				buttons[i].addActionListener(l);
			}
		}

	}


	ImageWithPoints imzone;
	ImageGetter imageGetter;
	DataDeliverer dataDeliverer;
	GaugeGauge hgg, vgg;
	MainServices myparent;

	Color hcolor, vcolor;
	boolean recogniserRunning = false;
	boolean statusLocked = false;
	Vector<Component> criticalComponents = new Vector<Component>();
	Vector<MenuItem> criticalMenuItems = new Vector<MenuItem>();

	TextArea results;
	Steeringstuff apanel;
	MagGlass magGlass;
	ScrollPane scrollp = new ScrollPane(ScrollPane.SCROLLBARS_ALWAYS);
	Panel helppanel = new Panel();
	Label rulerLabel;
	TextField fnamefield;
	MenuItem stopRecogniserItem;
	Checkbox sortX;
	RecogniserSettings recSettings=new RecogniserSettings();

	Rectangle bbox;
	String datasetName;
	String targetFName;
	int scale=3;
	static final long serialVersionUID = 20060308L;
	AffineTrafo transformation = null;

	
	// TODO: give DExtractor a seperate config class.
	// sourcebib, sourcepage, xAxisColor and yAxisColor should be
	// in there, I guess.  Possibly quite a few more...
	DExtractor(MainServices parent, Rectangle bbox, String sourcebib,
		String sourcepage, ImageGetter imageGetter,
		DataDeliverer dataDeliverer, Color xAxisColor, Color yAxisColor,
		int defaultScale)
	{
		this.datasetName = "# Graph from "+sourcebib+", page "+sourcepage;
		this.targetFName = sourcebib+"."+sourcepage;
		this.imageGetter = imageGetter;
		this.myparent = parent;
		this.bbox = bbox;
		this.dataDeliverer = dataDeliverer;
		this.hcolor = xAxisColor;
		this.vcolor = yAxisColor;
		this.scale = defaultScale;

		this.addComponentListener(this);
		addWindowListener(new WindowAdapter() { 
			public void windowClosing(WindowEvent event) 
			{ 
				closeWin();
      }
    });

		makeDialog();
		makeMenu();
		this.imzone.setImage(imageGetter.getImage(this.scale, bbox));
	}

	DExtractor(MainServices parent, Rectangle bbox, String sourcebib,
		String sourcepage, ImageGetter imageGetter,
		DataDeliverer dataDeliverer, Color xAxisColor, Color yAxisColor)
	{
		this(parent, bbox, sourcebib, sourcepage, imageGetter,
			dataDeliverer, xAxisColor, yAxisColor, 3);
	}

	DExtractor(MainServices parent, Rectangle bbox, String sourcebib,
		String sourcepage, ImageGetter imageGetter,
		DataDeliverer dataDeliverer)
	{
		this(parent, bbox, sourcebib, sourcepage, imageGetter,
			dataDeliverer, new Color(255, 0, 0), new Color(0, 0, 255), 3);
	}

	protected void makeMenu()
	{	
		MenuBar mb;
		MenuItem m;

		mb = new MenuBar();
		setMenuBar(mb);

		Menu fileMenu = new Menu("File");
		mb.add(fileMenu);
		m = new MenuItem("Show Data");
		fileMenu.add(m);
		this.criticalMenuItems.addElement(m);
		m = new MenuItem("Send Data");
		fileMenu.add(m);
		this.criticalMenuItems.addElement(m);
		m = new MenuItem("Save Data");
		fileMenu.add(m);
		this.criticalMenuItems.addElement(m);
		fileMenu.addSeparator();
		m = new MenuItem("Close");
		m.setShortcut(new MenuShortcut(KeyEvent.VK_W, false));
		m.setActionCommand("Close");
		fileMenu.add(m);
		fileMenu.addActionListener(this);

		Menu zoomMenu = new Menu("Zoom");
		mb.add(zoomMenu);
		m = new MenuItem("600 dpi");
		zoomMenu.add(m);
		this.criticalMenuItems.addElement(m);
		m = new MenuItem("300 dpi");
		zoomMenu.add(m);
		this.criticalMenuItems.addElement(m);
		m = new MenuItem("200 dpi");
		zoomMenu.add(m);
		this.criticalMenuItems.addElement(m);
		m = new MenuItem("100 dpi");
		zoomMenu.add(m);
		this.criticalMenuItems.addElement(m);
		m = new MenuItem("75 dpi");
		zoomMenu.add(m);
		this.criticalMenuItems.addElement(m);
		zoomMenu.addActionListener(this);

		Menu recMenu = new Menu("Recognize");
		mb.add(recMenu);
		m = new MenuItem("Trace a Line");
		m.setShortcut(new MenuShortcut(KeyEvent.VK_T, false));
		m.setActionCommand("Trace a Line");
		recMenu.add(m);
		this.criticalMenuItems.addElement(m);
		m = new MenuItem("Find Points");
		m.setShortcut(new MenuShortcut(KeyEvent.VK_B, false));
		m.setActionCommand("Find Points");
		recMenu.add(m);
		this.criticalMenuItems.addElement(m);
		m = new MenuItem("Automatic Axes");
		m.setShortcut(new MenuShortcut(KeyEvent.VK_A, false));
		m.setActionCommand("Automatic Axes");
		recMenu.add(m);
		this.criticalMenuItems.addElement(m);

		recMenu.addSeparator();
		m = new MenuItem("Recognizer Settings");
		m.setShortcut(new MenuShortcut(KeyEvent.VK_S, false));
		m.setActionCommand("Recognizer Settings");
		recMenu.add(m);

		stopRecogniserItem = new MenuItem("Stop Recognizer");
		recMenu.add(stopRecogniserItem);
		stopRecogniserItem.setEnabled(false);
		recMenu.addSeparator();
		recMenu.add(new MenuItem("Delete all Points"));
		recMenu.addActionListener(this);

		Menu helpMenu = new Menu("Help");
		mb.setHelpMenu(helpMenu);
		helpMenu.add(new MenuItem("About Dexter"));
		helpMenu.add(new MenuItem("Help"));
		helpMenu.addActionListener(this);
	}
		
	protected void makeDialog()
	{	
		GridBagLayout layout = new GridBagLayout();
		GridBagConstraints c = new GridBagConstraints();
		Button actbuts[] = {	
			new Button("Help"),
			new Button("Show Data"),
			new Button("Send Data"),
			new Button("Save Data"),
			new Button("Close"),
		};
		this.criticalComponents.addElement(actbuts[1]);
		this.criticalComponents.addElement(actbuts[2]);
		this.criticalComponents.addElement(actbuts[3]);

		Label lab;
		Panel zwp;

		setFont(new Font("Helvetica", Font.PLAIN, 10));

		apanel = new Steeringstuff(actbuts, 5, this);
		fnamefield = new TextField(targetFName, 20);

		this.setLayout(layout);
		this.magGlass = new MagGlass();
		this.hgg = new GaugeGauge("x", this.hcolor, false, this);
		this.vgg = new GaugeGauge("y", this.vcolor, true, this);
		this.results = new TextArea("", 9, 40);
		this.rulerLabel = new Label("        ");
		this.imzone = new ImageWithPoints(this, magGlass, recSettings);
		
		helppanel.setLayout(new GridBagLayout());
		c.fill = GridBagConstraints.BOTH;
		c.gridy = 0;
		((GridBagLayout)helppanel.getLayout()).setConstraints(apanel,c);
		helppanel.add(apanel);
		c.gridy = 1;
		((GridBagLayout)helppanel.getLayout()).setConstraints(vgg,c);
		helppanel.add(vgg);
		c.gridy = 2;
		((GridBagLayout)helppanel.getLayout()).setConstraints(magGlass,c);
		helppanel.add(magGlass);

		c.gridheight = 3;
		c.gridwidth = 1;
		c.gridx = 0;
		c.gridy = 0;
		c.fill = GridBagConstraints.NONE;
		c.weightx = 0;
		c.weighty = 1;
		c.anchor = GridBagConstraints.NORTH;
		layout.setConstraints(helppanel, c);
		add(helppanel);

		c.anchor = GridBagConstraints.CENTER;
		c.fill = GridBagConstraints.BOTH;
		c.gridwidth=GridBagConstraints.REMAINDER;
		c.gridheight=1;
		c.weightx = 1;
		c.weighty = 1;
		c.gridx = 1;
		c.gridy = 0;
		layout.setConstraints(this.scrollp, c);
		add(this.scrollp);
		scrollp.add(this.imzone);

		c.fill = GridBagConstraints.NONE;
		c.gridwidth = 1;
		c.weightx = 1;
		c.weighty = 0;
		c.gridx = 1;
		c.gridy = 1;
		layout.setConstraints(hgg, c);
		add(hgg);

		sortX = new Checkbox("Sorted", true);
		c.gridx = 2;
		add(sortX);

		zwp = new Panel();
		lab = new Label("File name:");
		zwp.add(lab);
		zwp.add(fnamefield);
		c.gridx = 3;
		add(zwp);

		c.fill = GridBagConstraints.HORIZONTAL;
		c.gridx = 1;
		c.gridwidth=GridBagConstraints.REMAINDER;
		c.gridy = 2;
		layout.setConstraints(results, c);
		add(results);

		c.fill = GridBagConstraints.HORIZONTAL;
		c.gridx = 1;
		c.gridwidth=GridBagConstraints.REMAINDER;
		c.gridy = 3;
		c.gridx = 0;
		rulerLabel.setBackground(Color.white);
		layout.setConstraints(rulerLabel, c);
		add(rulerLabel);
	}

	// the layout manager kept giving me rubbish here, so I fumble together
	// something that might be remotely sensible
	public Dimension getPreferredSize()
	{ 
		return new Dimension(
			Math.max(this.magGlass.getPreferredSize().width,
				this.apanel.getPreferredSize().width)+
				this.imzone.getPreferredSize().width+20,
			Math.max(this.helppanel.getPreferredSize().height+20,
				this.imzone.getPreferredSize().height+40)+
				this.hgg.getPreferredSize().height+
				this.results.getPreferredSize().height+40+
				this.rulerLabel.getPreferredSize().height);
	}

	public Dimension getMinimumSize()
	{	
		return new Dimension(600,400);
	}

	public synchronized void resizeToPreferredSize()
	{	
		Dimension pS=getPreferredSize();
		Dimension screenS=java.awt.Toolkit.getDefaultToolkit().getScreenSize();

		this.setSize(new Dimension(
			Math.min(screenS.width-40, Math.max(600, pS.width)),
			Math.min(screenS.height-40, Math.max(400, pS.height))));
		validate();
		repaint();
		// Incredibly nasty hack -- how do I keep the titlebar from
		// vanishing on some platforms sanely?  And why is it that
		// setLocation(getLocation) is not a no-op?
		if (this.isShowing() && this.getLocation().y<10) {
			this.setLocation(20, 40);
		}
	}

	private void newimagescale(int newscale)
	{	
		Image im;

		imzone.newScale(scale, newscale);
		im = imageGetter.getImage(newscale, bbox);
		imzone.setImage(im);
		magGlass.setImage(im);
		scale = newscale;
		this.resizeToPreferredSize();
	}
	
	public void closeWin()
	{
		setVisible(false);
		if (recSettings.isShowing()) {
			recSettings.close();
		}
		myparent.childClosed();
	}

	public void actionPerformed(ActionEvent e)
	{	
		if (e.getActionCommand()=="Close") {
			closeWin();
		}
		if (e.getActionCommand()=="75 dpi") {	
			newimagescale(8); 
		}
		if (e.getActionCommand()=="100 dpi") { 
			newimagescale(6); 
		}
		if (e.getActionCommand()=="200 dpi") {	
			newimagescale(3); 
		}
		if (e.getActionCommand()=="300 dpi") {
			newimagescale(2); 
		}
		if (e.getActionCommand()=="600 dpi") {	
			newimagescale(1); 
		}

		if ((e.getActionCommand()=="Show Data") ||
			(e.getActionCommand()=="Send Data") ||
			(e.getActionCommand()=="Save Data")) {	
			try { 
				makedata();
			} catch (CantComputeException exc) {	
				return; 
			}

			if (e.getActionCommand()=="Save Data") {
				dataDeliverer.deliver(results.getText(), fnamefield.getText(), 1);
			}
			if (e.getActionCommand()=="Send Data") {
				dataDeliverer.deliver(results.getText(), fnamefield.getText(), 0);
			}
		}

		if (e.getActionCommand()=="Help")
			myparent.showHelp();

		if (e.getActionCommand()=="About Dexter")
		{	new AlertBox(this, "About Dexter", "Dexter -- Data extraction applet"+
				"\nRelease 0.5a"+
				"\nA part of the NASA Astrophysics Data System\n"+
				"http://adswww.harvard.edu\n");
		}

		if (e.getActionCommand()=="Trace a Line") {
			this.startRecogniser("LineTracer");
		}
		if (e.getActionCommand()=="Find Points") {
			this.startRecogniser("PointFinder");
		}
		if (e.getActionCommand()=="Automatic Axes") {
			this.startRecogniser("AxisFinder");
		}

		if (e.getActionCommand()=="Recognizer Settings") {
			this.recSettings.open();
		}
		if (e.getActionCommand()=="Stop Recognizer") {
			this.stopRecogniser();
		}

		if (e.getActionCommand()=="Delete all Points") {
			this.imzone.delAllPoints();
		}
	}
	
	protected void enableCritical(boolean enable)
	{	
		for (Iterator<Component> it=this.criticalComponents.iterator();
				it.hasNext();) {
			it.next().setEnabled(enable);
		}
		for (Iterator<MenuItem> it=this.criticalMenuItems.iterator();
				it.hasNext();) {
			it.next().setEnabled(enable);
		}
	}

	protected synchronized void startRecogniser(String recName)
	{
		try {
			this.imzone.startRecogniser(recName);
		} catch (Exception ex) {
			new AlertBox(this, "Dexter Error Message", ex.getMessage());
			return;
		}
		this.recogniserRunning = true;
		this.enableCritical(false);
		this.stopRecogniserItem.setEnabled(true);
	}


	synchronized void recogniserStopped()
	{	
		this.recogniserRunning = false;
		this.releaseStatusLine();
		this.stopRecogniserItem.setEnabled(false);
		this.enableCritical(true);
	}


	synchronized void stopRecogniser()
	{	
			imzone.stopRecogniser();
	}


	// Compute the transformation -- zerox and zeroy are the zero points
	// of the axes in pixel coordinates, xaxis and yaxis are unit vectors
	// of these axes, islog[i] is true when the respective axis is 
	// logarithmic
	protected void computeTransform() throws CantComputeException
	{	
		try {	
			this.transformation = new AffineTrafo(
				imzone.hgauge.start, imzone.vgauge.start,
				imzone.hgauge.end, imzone.vgauge.end,
				this.hgg.getstart(), this.hgg.getend(),
				this.vgg.getstart(), this.vgg.getend(),
				this.hgg.isLogAxis(), this.vgg.isLogAxis());
		} catch (java.lang.NumberFormatException e) {	
			this.transformation = null;
			throw new CantComputeException("You must fill out both"+
				" horizontal and vertical\n gauge number fields with sensible values");
		} catch (NullPointerException e) {
			this.transformation = null;
			throw new CantComputeException("Both axes must be defined"+
				" before value\nreadout is possible.\n  Click and drag to"+
				" define axes.");
		} catch (MissingData e) {
			this.transformation = null;
			throw new CantComputeException(e.getMessage());
		}
	}


	private void makedata_orth() 
	{	
		Datapoint points[];
		boolean needVertBars=false, needHorizBars=false;
	
		// Update transform and bail out if we don't have enough
		// data to transform
		try {
			this.computeTransform();
		} catch (CantComputeException e) {
			 new AlertBox(this, "Dexter Error", e.getMessage());
			 return;
		}

		points = this.imzone.getPoints();
		// sort for abscissa value
		if (this.sortX.getState()) {
			QuickSort.getInstance().sort(
				points, 0, points.length-1, points[0]);
		}

		// check if we need to give error bars
		for (int i=0; i<points.length; i++) {
			Datapoint pt = points[i];
			needHorizBars |= pt.hasHorizErrBars();
			needVertBars |= pt.hasVertErrBars();
		}

		results.setText(datasetName+"\n");
		int sigFigX = this.hgg.computeUsefulSignificantDigits();
		int sigFigY = this.vgg.computeUsefulSignificantDigits();
		for (int i=0; i<points.length; i++) {
			results.append(points[i].getRepr(
				needHorizBars, needVertBars, 
				sigFigX, sigFigY));
		}
	}

	public void makedata()
	{	
		makedata_orth();
	}

	/**
	 * returns logical coordinates for the screen coordinates coo if a
	 * logical coordinate system is already defined, a DoublePoint for coo
	 * otherwise.
	 */
	public DoublePoint transformPhysicalToLogical(Point coo)
	{
		DoublePoint pt = new DoublePoint(coo);
		if (this.transformation!=null) {
			pt = this.transformation.transformPhysicalToLogical(coo);
		}
		return pt;
	}

	public void displayMousePos(Point mousePos)
	{	
		if (statusLocked) {
			return;
		}
		DoublePoint pt = this.transformPhysicalToLogical(mousePos);
		String posTx = Fmt.fmt(pt.getX(), 4, 
				this.hgg.computeUsefulSignificantDigits())+" / "+
				Fmt.fmt(pt.getY(), 4, this.vgg.computeUsefulSignificantDigits());
		if (recogniserRunning) {
			posTx = posTx+" Recog. running";
		}
		rulerLabel.setText(posTx);
	}

	public boolean allocStatusLine(String msg)
	{	
		if (statusLocked) {
			return false;
		}
		statusLocked = true;
		rulerLabel.setText(msg);
		return true;
	}


	public void releaseStatusLine()
	{
		statusLocked = false;
		rulerLabel.setText("");
	}

	public void componentHidden(ComponentEvent e) {}
	public void componentMoved(ComponentEvent e) {}
	public void componentResized(ComponentEvent e) {}
	public void componentShown(ComponentEvent e) {}
}
// vi:ts=2:
