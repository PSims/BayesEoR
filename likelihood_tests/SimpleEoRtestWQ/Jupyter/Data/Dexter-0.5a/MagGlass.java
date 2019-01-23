// A Magnifying Glass
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
import java.awt.image.*;
import java.awt.event.*;

public class MagGlass extends Canvas implements ImageObserver
{	
	Image im;
	int sx0=0, sy0=0;
	int sx=21, sy=21;
	//int sx=121, sy=121;
	int imscale=3; //had better be odd
	int fullx=sx*imscale, fully=sy*imscale;
	int halfx=fullx/2, halfy=fully/2;
	boolean active=false;
	Image bufferImage=null; // this is a hack necessary to provide 
		// reasonable performance -- it seems quite a few runtimes 
		// don't like scaling and cropping in one drawImage
	Graphics bufGraphics=null;
	static final long serialVersionUID=20060308L;

	public MagGlass()
	{ 
		this(null);
	}


	public MagGlass(Image im)
	{
		this.im = im;
		addMouseListener(m);
	}


	public void setImage(Image im)
	{
		this.im = im;
	}


	public synchronized boolean imageUpdate(Image targIm,int infoflags,
		int x,int y,int width, int height)
	{	
		if (targIm!=im)
			return false;
		if ((infoflags&ImageObserver.ALLBITS)!=0)
		{	repaint();
			return false;
		}
		return true;
	}


	public synchronized void setCoords(Point p)
	{
		sx0 = Math.max(p.x-sx/2,0);
		sy0 = Math.max(p.y-sy/2,0);
		if (!active)
			return;
		if (bufferImage==null)
		{	bufferImage = createImage(sx,sy);
			if (bufferImage!=null)
				bufGraphics = bufferImage.getGraphics();
		}
		if (bufGraphics==null)
			return;
		bufGraphics.drawLine(0,0,sx,sy);
		bufGraphics.drawImage(im,0,0,sx,sy,sx0,sy0,sx0+sx,sy0+sy,this);
		repaint();
	}
	
	
	public synchronized void paint(Graphics g)
	{	
		if (!active || im==null)
		{	g.setColor(getBackground());
			g.fillRect(0,0,fullx,fully);

			g.setColor(getForeground());
			int icD=Math.min(fullx,fully)/3;
			Point ic0=new Point(icD,icD);
			g.drawOval(ic0.x-icD/2,ic0.y-icD/2,icD,icD);
			g.drawOval(ic0.x-icD/2,ic0.y-icD/2,icD-1,icD-1);
			int sx = (int)Math.round(ic0.x+icD/2/1.41);
			int sy = (int)Math.round(ic0.y+icD/2/1.41);
			int ex = Math.round(ic0.x+icD);
			int ey = Math.round(ic0.y+icD);
			g.drawLine(sx,sy,ex,ey);
			g.drawLine(sx+1,sy-1,ex+1,ey-1);
			g.drawLine(sx-1,sy+1,ex-1,ey+1);
			return;
		}
		if (im!=null)
		{ //ImageProducer ip=new FilteredImageSource(im.getSource(),
			//	new CropImageFilter(sx0,sy0,sx,sy));
			//g.drawImage(createImage(ip).getScaledInstance(
			//	fullx,fully,Image.SCALE_FAST),0,0,null);
			if (bufferImage!=null)
				g.drawImage(bufferImage,0,0,fullx,fully,0,0,sx,sy,null);
			g.setColor(Color.green);
			g.drawLine(halfx,0,halfx,fully);
			g.drawLine(0,halfy,fullx,halfy);
		}
	}


	public synchronized void update(Graphics g)
	{
		paint(g);
	}


	public Dimension getPreferredSize()
	{
		return new Dimension(fullx, fully);
	}


	public Dimension getMaximumSize()
	{
		return getPreferredSize();
	}


	public Dimension getMinimumSize()
	{
		return getPreferredSize();
	}

	MouseAdapter m=new MouseAdapter()
	{ public void mouseClicked(MouseEvent e)
		{
			active = !active;
			repaint();
		}
	};

}
