// A class that gets the extracted data to the user.
//
// Copyright (c) 2000 Markus Demleitner <mdemleitner@head-cfa.harvard.edu>
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
import java.net.URL;
import java.net.MalformedURLException;
import java.net.URLConnection;
import java.io.*;

class AppletDataDeliverer implements DataDeliverer
{	
	Dexter parent;

	public AppletDataDeliverer(Dexter parent)
	{
		this.parent = parent;
	}

	private Thread getSendThread(final String pipepart, final String toSend,
			final int option) throws MalformedURLException {
		final String receivescript = this.parent.receiverURI+"?";
		final URL saveurl = new URL(receivescript+pipepart);

		return new Thread() {
			public void run() {

				PrintWriter out = null;
				URLConnection savec = null;
				URL saveurl = null;
				try {
					saveurl = new URL(receivescript+pipepart);
				} catch (MalformedURLException me) {
					System.out.println("Malformed URL, giving up");
									return;
				}
				try {	
					savec = saveurl.openConnection();
					savec.setDoOutput(true);
					out = new PrintWriter(savec.getOutputStream());
				} catch (IOException me) {	
					System.out.println("Cannot open output, giving up.");
					return;
				}
				if (option==1) {
					out.println("content-disposition: attachment; filename=data.txt");
					out.println("content-type: application/octet-stream\n\n"+
						toSend);
				} else {
					out.println("content-type: text/plain\n\n"+
						toSend);
				}
				out.close();

				// we don't actually want to know what's coming back (yet),
				// but everyone's much happier if we at least properly throw
				// away anything the server's giving us back.
				try {	
					BufferedReader in = new BufferedReader(
						new InputStreamReader(savec.getInputStream()));

					while ((in.readLine()) != null) {
					}
					in.close();
				} catch (java.io.IOException e) {
				}
			}
		};
	}

  /**
	 * makes the browser retrieve whatever results are written in sendThread.
	 *
	 * @param pipepart -- a cookie shared with the send thread
	 * @param dest -- the window id the browser is to use to display the data
	 * @param fname -- the suggested file name for the browser
	 */
	private Thread getRecvThread(final String pipepart, 
			final String dest, final String fname) {
		final String sendscript = this.parent.senderURI+
			"/"+fname.replace('&','+')+"?";
		return new Thread() {	
			public void run() {	
				try { 
					parent.getAppletContext().showDocument(
						new URL(sendscript+pipepart), dest);
				} catch (MalformedURLException e) {
				}
			}
		};
	}

	/**
	 * arranges for the content of toSend to be delivered to the current
	 * client.
	 *
	 * @param toSend the data to send
	 * @param fname a file name suggested to the user
	 * @param option 0 if content is to be declared text/plain, 1 for
	 *        application/octet-stream
	 *
	 * Since we're generally running unsigned, the host has to help by
	 * accepting an incoming data stream and funneling it out again.
	 * Two perl scripts need to be present on the host to pull that off,
	 * and I need a second frame that acts as a target for the download.
	 */

	public void deliver(String toSend, String fname, int option)
	{	
		final String pipepart = "pipename="+"dp_pipe"+
			Integer.toString((int)(Math.random()*Integer.MAX_VALUE))+
			Integer.toString((int)(Math.random()*Integer.MAX_VALUE));

		final String dest="dp_save";
		this.getRecvThread(pipepart, dest, fname).start();

		try {
			this.getSendThread(pipepart, toSend, option).start();
		} catch (MalformedURLException e) {
			// XXX do something about it :-)
		}
	}
}

// vi:ts=2:
