// Copyright (c) 2000, 2004 Markus Demleitner <msdemlei@cl.uni-heidelberg.de>
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
import java.awt.event.*;
import java.awt.image.*;
import java.io.*;
import java.net.URL;
import java.net.MalformedURLException;


class Gaucho_w extends Frame implements MainServices, ActionListener,
	ComponentListener
{	
	DExtractor worker;
	String imageURL;
	int scale;
	ScrollImScrollBar sel=null;
	DExtractor window=null;
	String sourcebib,sourcepage;
	boolean onIndexedDisplay=false;
	ScriptImageGetter imageGetter;
	PlainDataDeliverer dataDeliverer=new PlainDataDeliverer(this);
	static final long serialVersionUID=20060308L;

	public Gaucho_w(String args[])
	{ 
		this.addWindowListener(new WindowAdapter() {
				public void windowClosing(WindowEvent e) {
					System.exit(0);
				}
			});

		this.addComponentListener(this);

		ColorModel colormodel = Toolkit.getDefaultToolkit().getColorModel();
		if (colormodel instanceof IndexColorModel) {
			this.onIndexedDisplay = true;
		}

		this.setTitle("Gaucho");
		this.makeMenu();

		this.imageGetter = new ScriptImageGetter(this, args[0]);
		this.sourcebib = args[0];
		this.sourcepage = "1";
		this.scale = 8;

		this.makeContents();
		this.notifyChangedSize();
	}

	protected void makeContents()
	{
		Image fullpg;

		fullpg = imageGetter.getImage(this.scale);
		this.sel = new ScrollImScrollBar(this, fullpg, 
			this.onIndexedDisplay, false);
		this.setLayout(new GridLayout(1, 1));
		this.add(this.sel);
		this.validate();
		this.sel.doLayout();
	}

	protected void makeMenu()
	{
		MenuBar mb;
		MenuItem m;

		mb = new MenuBar();
		this.setMenuBar(mb);
		Menu fileMenu = new Menu("File");
		fileMenu.addActionListener(this);
		mb.add(fileMenu);
		m = new MenuItem("Quit");
		m.setShortcut(new MenuShortcut(KeyEvent.VK_Q, false));
		m.setActionCommand("Quit");
		fileMenu.add(m);
	}

	public void actionPerformed(ActionEvent e)
	{
		if (e.getActionCommand()=="Quit") {
			this.closeChild();
			System.exit(0);
		}
	}

	protected void closeChild()
	{	
		if (this.window!=null)	{	
			if (this.window.isShowing()) {
				this.window.closeWin();
			}
			this.window = null;
		}
	}
	
	public void notifySelection(Rectangle bbox)
	{
		if (bbox.width<50 || bbox.height<50)
			return;
		this.closeChild();
		bbox.x *= scale;
		bbox.y *= scale;
		bbox.width *= scale;
		bbox.height *= scale;
		try {
			this.window = new DExtractor(this, bbox, sourcebib, sourcepage, 
				imageGetter, dataDeliverer);
		} catch (NullPointerException ex) {
			new AlertBox(this, "Gaucho Error Message",
				"Image retrieval failed.  Check convert.log.");
			return;
		}
		this.window.setTitle("Dexter");
		this.window.pack();
		this.window.setVisible(true);
	}

	// this is a callback for the child
	public void childClosed()
	{
		window = null;
	}

	public void showHelp()
	{	
	}

	public Dimension getPreferredSize()
	{
		if (this.sel==null) {
			return new Dimension(400, 400);
		} else {
			Dimension ps=this.sel.getPreferredSize();
			Insets insets=this.getInsets();
			return new Dimension((int)(ps.getWidth()+insets.left+insets.right),
				(int)(ps.getHeight()+insets.top+insets.bottom));
		}
	}

	public void resizeToPreferredSize()
	{	
		Dimension ps=this.getPreferredSize();
		Dimension screens=java.awt.Toolkit.getDefaultToolkit().getScreenSize();
		Dimension newSize=new Dimension(
			Math.min(screens.width-40, Math.max(400, ps.width)),
			Math.min(screens.height-40, Math.max(400, ps.height)));

		setSize(newSize);
		validate();
		repaint();
		// Incredibly nasty hack -- how do I keep the title bar from
		// vanishing on some platforms sanely?
		if (this.isShowing() && this.getLocation().y<10) {
			this.setLocation(20, 40);
		}
	}

	public void notifyChangedSize()
	{
		this.resizeToPreferredSize();
	}

	public void componentHidden(ComponentEvent e) {}
	public void componentMoved(ComponentEvent e) {}
	public void componentResized(ComponentEvent e) {}
	public void componentShown(ComponentEvent e) 
	{
		this.notifyChangedSize();
	}
}


public class Gaucho
{
	public static void main(String[] args)
	{	Gaucho_w g;

		g = new Gaucho_w(args);
		g.pack();
		g.setVisible(true);
	}
}

// vim:ts=2
