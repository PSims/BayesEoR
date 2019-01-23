// An image getter that puts all the complexity into a shell scipt
//
// Copyright (c) 2003 Markus Demleitner <msdemlei@cl.uni-heidelberg.de>
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
import java.io.*;
import java.util.*;
import java.lang.*;

public class ScriptImageGetter implements ImageGetter
{
	Object parent;
	String fname;

	private class IoChunk
	{
		public byte[] contents;
		public int len;
		
		public IoChunk(byte[] contents, int len)
		{
			this.contents = new byte[len];
			for (int i=0; i<len; i++) {
				this.contents[i] = contents[i];
			}
		}
	}

	public ScriptImageGetter(Object parent, String fname)
	{
		this.parent = parent;
		this.fname = fname;
	}

	private Vector<IoChunk> getOutputChunksFromProcess(Process p) 
		throws IOException
	{
		final int bufferSize = 10240;
		byte[] buffer=new byte[bufferSize];
		int readBytes;
		Vector<IoChunk> chunkVec = new Vector<IoChunk>();
	  BufferedInputStream reader = 
			new BufferedInputStream(p.getInputStream());
  
	  while ((readBytes=reader.read(buffer, 0, bufferSize))!=-1) {
	    chunkVec.add(new IoChunk(buffer, readBytes));
   	}
		return chunkVec;
	}

	private int getTotalSizeOfChunks(Vector<IoChunk> chunks)
	{
		int totalSize = 0;
		
		for (Enumeration<IoChunk> chunkEnum = chunks.elements();
			chunkEnum.hasMoreElements();) {
			totalSize += (chunkEnum.nextElement()).contents.length; 
		}
		return totalSize;
	}

	private byte[] joinChunksToByteArr(Vector<IoChunk> chunks)
	{
		byte[] joined = new byte[getTotalSizeOfChunks(chunks)];
		int storeInd = 0;

		for (Enumeration<IoChunk> chunkEnum = chunks.elements();
			chunkEnum.hasMoreElements();) {
			byte[] stuff = (chunkEnum.nextElement()).contents; 
			for (int i=0; i<stuff.length; i++) {
				joined[storeInd++] = stuff[i];
			}
		}
		return joined;
	}

	public byte[] getCommandOutput(String[] commandAndArgs)
	{
		Vector<IoChunk> chunks;

		try {
		  Process p = Runtime.getRuntime().exec(commandAndArgs);
			chunks = getOutputChunksFromProcess(p);

			if (p.waitFor()!=0) {
				throw new IOException("Image getter script failed");
			}
		  p.destroy();
		} catch (Exception ex) {
			return null;
		}


		return joinChunksToByteArr(chunks);
	}

	public Image getImage(int scale)
	{
		Image im;
		String cmds[] = {
			"dexter_getImage",
			"-s",
			(new Integer(scale)).toString(),
			this.fname
		};

		return Toolkit.getDefaultToolkit().createImage(
			getCommandOutput(cmds));
	}

	public Image getImage(int scale, Rectangle bbox)
	{ 
		Image im;
		String cmds[] = {
			"dexter_getImage",
			"-s",
			(new Integer(scale)).toString(),
			"-b",
			bbox.x+","+bbox.y+","+bbox.width+","+bbox.height,
			this.fname
		};

		System.out.println("Acquiring image.");
		return Toolkit.getDefaultToolkit().createImage(
			getCommandOutput(cmds));
	}
}
