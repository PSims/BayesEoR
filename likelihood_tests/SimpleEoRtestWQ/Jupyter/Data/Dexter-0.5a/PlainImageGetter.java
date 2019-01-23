// A trivial and slightly braindamaged image getter for debug purposes.
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
//tabsize=2

import java.awt.*;
import java.awt.image.ImageObserver;

public class PlainImageGetter 
	implements ImageGetter, ImageObserver
{	
	Object parent;
	String fname;

	public PlainImageGetter(Object parent, String fname)
	{
		this.parent = parent;
		this.fname = fname;
	}

	public Image getImage(int scale)
	{	
		Image im;

		im = Toolkit.getDefaultToolkit().getImage(fname);
		if (im.getWidth(this)<0) {
			try { // need to do this to make sure I know the unscaled size
				MediaTracker mt=new MediaTracker((Component)parent);
				mt.addImage(im,1);
				mt.waitForAll();
			} catch (InterruptedException e) {
				System.out.println("Interrupt while waiting for image.\n");
			}
		}
		return im.getScaledInstance(im.getWidth(this)/scale*3, -1,
			java.awt.Image.SCALE_DEFAULT);
	}

	public Image getImage(int scale, Rectangle bbox)
	{ 
		return getImage(scale);
	}

	public synchronized boolean imageUpdate(Image im, int a, int b, int c, int d, int e)
	{
		return false;
	}
}
