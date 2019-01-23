// The main applet code
//
// Copyright (c) 2000, 2003 Markus Demleitner <msdemlei@cl.uni-heidelberg.de>
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

import java.applet.*;
import java.awt.*;
import java.awt.image.*;
import java.net.URL;
import java.net.MalformedURLException;
import java.awt.Color;


public class Dexter extends Applet implements MainServices
{	
	DExtractor worker;
	String imageURL;
	int scale, exScale;
	ScrollImScrollBar sel=null;
	DExtractor window=null;
	String sourcebib,sourcepage;
	boolean onIndexedDisplay=false;
	AppletImageGetter imageGetter;
	AppletDataDeliverer dataDeliverer=new AppletDataDeliverer(this);
	Color xAxisColor, yAxisColor;
	static final long serialVersionUID=20060308L;
	public String senderURI, receiverURI;
	boolean skipSelection;

	public void init()
	{ 
		Image fullpg;
		ColorModel colormodel = Toolkit.getDefaultToolkit().getColorModel();

		if (colormodel instanceof IndexColorModel) {	
			onIndexedDisplay = true;
		}

		try {	
			this.scale = Integer.parseInt(getParameter("DEFAULTSCALE"));
		}	catch(NumberFormatException e) {	
			this.scale = 8;
		}

		try {	
			this.exScale = Integer.parseInt(getParameter("EXDEFAULTSCALE"));
		}	catch(NumberFormatException e) {	
			this.exScale = 3;
		}

		imageURL = getParameter("SOURCEIMAGE");
		imageGetter = new AppletImageGetter(this, imageURL);
		fullpg = imageGetter.getImage(scale);

		try {	
			sourcebib = getParameter("BIBCODE");
		} catch(NumberFormatException e) {	
			sourcebib = "unknown";
		}

		try {	
			sourcepage = getParameter("PAGE");
		} catch(NumberFormatException e) {	
			sourcepage = "unknown";
		}

		try {
			this.xAxisColor = this.makeColor(
				this.getParameter("XAXISCOLOR"));
		} catch(Exception e) {
			this.xAxisColor = new Color(200, 0, 0);
		}

		try {
			this.yAxisColor = this.makeColor(
				this.getParameter("YAXISCOLOR"));
		} catch(Exception e) {
			this.yAxisColor = new Color(50, 50, 255);
		}

		String defaultURLStart = "http://"+this.getCodeBase().getHost();
		this.receiverURI = this.getParameter("RECEIVERURL");
		if (this.receiverURI==null) {
			this.receiverURI = defaultURLStart+"/cgi-bin/dp_receive.pl";
		}

		this.senderURI = this.getParameter("SENDERURL");
		if (this.senderURI==null) {
			this.senderURI = defaultURLStart+"/cgi-bin/dp_send.pl";
		}

		String rawSkipSel = this.getParameter("SKIPSELECTION");
		if (rawSkipSel.compareTo("True")==0) {
			this.skipSelection = true;
		} else {
			this.skipSelection = false;
		}

		try {
			this.sel = new ScrollImScrollBar(this, fullpg, onIndexedDisplay,
				this.skipSelection);
		} catch (Exception e) {
			System.out.println("Yikes -- no image.  I'm dying.");
		}
		setLayout(new GridLayout(1, 1));
	}

	public void start()
	{	
		if (this.sel==null) {
			Graphics g=getGraphics();
			if (g!=null) {	
				try {	
					g.setColor(Color.black);
					g.drawString("Could not connect to image source.\n"+
						"  Check your java console.", 10, 10);
				} finally {	
					g.dispose();
				}
			}
		} else {
			this.add(this.sel);
			this.validate();
			this.sel.doLayout();
		}
	}

	protected void closeWindow()
	{	
		if (window!=null)	{	
			if (window.isShowing()) {
				window.closeWin();
			}
			window = null;
		}
	}

	public void stop()
	{	
		if (sel!=null) {
			remove(sel);
		}
		closeWindow();
	}

	public void destroy()
	{	
	}

	public synchronized void paint(Graphics g)
	{
	}

	public void openWindow(Rectangle bbox)
	{
		try {
			this.window = new DExtractor(this, bbox, this.sourcebib, 
				this.sourcepage, this.imageGetter, this.dataDeliverer, 
				this.xAxisColor, this.yAxisColor, this.exScale);
			window.setTitle("Dexter");
			window.pack();
			window.setVisible(true);
		} catch (SecurityException e) {
			new AlertBox(this.getEnclosingFrame(), 
				"Dexter Error", "I am not allowed to"+
				" open a window\non this machine.\n"+
				" Please review the security options\n"+
				" for your virtual machine and allow"+
				" me to open windows.  Thank you.");
		}
	}

	public void notifySelection(Rectangle bbox)
	{
		if (bbox!=null && (bbox.width<50 || bbox.height<50))
			return;
		this.closeWindow();
		if (bbox!=null) {
			bbox.x *= this.scale;
			bbox.y *= this.scale;
			bbox.width *= this.scale;
			bbox.height *= this.scale;
		}
		this.openWindow(bbox);
	}

	protected Frame getEnclosingFrame()
	{
		Component parentComponent=this.getParent();

		while (parentComponent!=null && 
				!(parentComponent instanceof Frame)) {
			parentComponent = parentComponent.getParent();
		}
		return (Frame)parentComponent;
	}

	// this is a callback for the child
	public void childClosed()
	{
		window = null;
	}


	public void notifyChangedSize()
	{
	}


	public void showHelp()
	{	
		String helpURL=this.getParameter("HELPURL");

		if (helpURL==null) {
			helpURL = getDocumentBase()+"/Dexterhelp.html";
		}
		try {	
			getAppletContext().showDocument(new URL(helpURL), "_new");
		} catch (MalformedURLException ex) {}
	}

	public String getAppletInfo()
	{
		return "Dexter by Markus Demleitner (ads@cfa.harvard.edu), "+
			"a little tool to extract data from graphs";
	}

	public String[][] getParameterInfo()
	{	
		String[][] info = {
			{"DEFAULTSCALE",	"int",	
				"initial scale in selection window"},
			{"EXDEFAULTSCALE",	"int",	
				"initial scale in extraction window"},
			{"SOURCEIMAGE", "URL",	
				"URL of CGI that produces the image"},
			{"BIBCODE",	"String",	
				"Bibcode of article (for file name)"},
			{"PAGE",	"int",	
				"Page within article (for file name)"},
			{"XAXISCOLOR", "String", 
				"Color of x axis marker in rrggbb "+
				"format (hex chars)"},
			{"YAXISCOLOR", "String", 
				"Color of y axis marker in rrggbb "+
				"format (hex chars)"},
			{"SENDERURL", "String", 
				"Base URL of the result sender on host"},
			{"RECEIVERURL", "String", 
				"Base URL of the result receiver on host"},
			{"HELPURL", "String", 
				"URL to Dexter's help file"},
			{"SKIPSELECTION", "boolean", 
				"Open extraction window immediately with full image"},
		};

		return info;
	}

	private Color makeColor(String colorSpec)
	// returns an awt Color from an X11-Style color spec (rrggbb)
	{
		return new Color(
			Integer.parseInt(colorSpec.substring(0, 2), 16),
			Integer.parseInt(colorSpec.substring(2, 4), 16),
			Integer.parseInt(colorSpec.substring(4, 6), 16));
	}
}

// vim:ts=2:
